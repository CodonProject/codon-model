from codon import *
from codon.block.embedding import BasicEmbedding
from codon.ops import (
    AttentionOutput,
    apply_attention
)
from codon.ops.attn_cuda import (
    triton_gqa_forward,
    HAS_TRITON_GQA
)
from codon.model.cache import (
    BasicLayerCache,
    KVLayerCache
)
from codon.block.attention.base import BasicAttention


class MultiHeadAttention(BasicAttention):
    '''
    Pure Multi-Head Attention (MHA / GQA) module.

    Supports standard MHA, Grouped Query Attention (GQA), QK Normalization,
    an optional output gate, RoPE injection via an external position embedding,
    and Triton GQA kernels for CUDA Prefill acceleration.

    This is the mechanism-split successor of the original `MultiHeadAttention`
    (now `MultiHeadAttentionLegacy`), which bundled MHA/MLA/HCA/CSA in one class.

    Attributes:
        q_proj (nn.Linear): Query projection.
        k_proj (nn.Linear): Key projection (GQA-compressed).
        v_proj (nn.Linear): Value projection (GQA-compressed).
        o_proj (nn.Linear): Output projection.
        q_norm (nn.RMSNorm, optional): Per-head Q normalization.
        k_norm (nn.RMSNorm, optional): Per-head K normalization.
        g_proj (nn.Linear, optional): Output gating projection.
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        use_qk_norm: Optional[bool] = True,
        use_gate: Optional[bool] = False,
        dropout: Optional[float] = 0.1,
        bias: Optional[bool] = True,
        is_causal: Optional[bool] = True,
        rope_dim: Optional[int] = 0,
        use_triton: bool = False,
    ):
        '''
        Initialize a standard MHA / GQA attention layer.

        Args:
            hidden_size (int): Hidden dimension.
            num_heads (int): Number of attention heads.
            num_kv_heads (int, optional): Number of KV heads for GQA.
                                          If None, defaults to num_heads (MHA).
            use_qk_norm (bool, optional): Apply per-head RMSNorm to Q and K.
                                          Defaults to True.
            use_gate (bool, optional): Apply a sigmoid gate on the output.
                                       Defaults to False.
            dropout (float, optional): Attention dropout probability. Defaults to 0.1.
            bias (bool, optional): Whether the linear projections have bias.
                                   Defaults to True.
            is_causal (bool, optional): Whether to apply a causal mask.
                                        Defaults to True (decoder).
            rope_dim (int, optional): If > 0, split the last `rope_dim` dims of each
                                      head and inject RoPE only there (partial rotary).
                                      Defaults to 0 (full-head RoPE via position_emb).
            use_triton (bool, optional): Enable Triton GQA kernels on CUDA prefill.
                                         Defaults to False.
        '''
        # 几何推导（head_dim/kv_dim/num_kv_queries 及整除断言）由 BasicAttention 完成
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate
        self.dropout = dropout
        self.is_causal = is_causal
        self.rope_dim = rope_dim
        self.use_triton = use_triton

        # 1. Query / Key / Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)

        # 2. QK normalization (per head)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        # 3. Optional output gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # 4. Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    @classmethod
    def cache_type(cls) -> type:
        '''标准 MHA/GQA 使用 KV 缓存。'''
        return KVLayerCache

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: BasicEmbedding = None,
        embedding_start: int = 0,
        embedding_pos: torch.Tensor = None,
        past_key_value: Optional[BasicLayerCache] = None,
    ) -> AttentionOutput:
        '''
        Standard MHA / GQA forward pass with optional KV cache and RoPE.

        Args:
            hidden_states (torch.Tensor): Input hidden states [B, L, H].
            kv_states (torch.Tensor, optional): Key/value source states.
                                                If None, uses hidden_states.
            attention_mask (torch.Tensor, optional): Attention mask.
            output_attentions (bool, optional): Return attention weights.
            position_emb (BasicEmbedding, optional): Positional embedding module.
            embedding_start (int, optional): Starting position index.
            embedding_pos (torch.Tensor, optional): Explicit position indices.
            past_key_value (BasicLayerCache, optional): KV cache (KVLayerCache).

        Returns:
            AttentionOutput: Output tensor, optional weights, and updated cache.
        '''
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))

        # Triton kernel eligibility: CUDA prefill (q_len > 1), no attn-weights request
        can_use_triton = self.use_triton and hidden_states.is_cuda and q_len > 1 and not output_attentions

        # ---- 1. Query ----
        q = self.reshape_q(self.q_proj(hidden_states), q_len, batch_size)

        if self.rope_dim > 0:
            # Partial rotary: normalize the non-rotary part, rotate the last rope_dim
            q_c, q_p = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
            if self.use_qk_norm:
                q_c = self.q_norm(q_c)
            if position_emb is not None:
                q_p = position_emb(q_p, start_pos=embedding_start, positions=embedding_pos)
            q = torch.cat([q_c, q_p], dim=-1)
        else:
            if self.use_qk_norm:
                q = self.q_norm(q)
            if position_emb is not None:
                q = position_emb(q, start_pos=embedding_start, positions=embedding_pos)

        # ---- 2. Key / Value (standard MHA / GQA) ----
        use_cache = isinstance(past_key_value, KVLayerCache)
        k = self.reshape_kv(self.k_proj(kv_states), kv_len_input, batch_size)
        v = self.reshape_kv(self.v_proj(kv_states), kv_len_input, batch_size)

        current_key_value = None
        if self.rope_dim > 0:
            k_c, k_p = k.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
            if self.use_qk_norm:
                k_c = self.k_norm(k_c)
            if position_emb is not None:
                k_p = position_emb(k_p, start_pos=embedding_start, positions=embedding_pos)
            k = torch.cat([k_c, k_p], dim=-1)
        else:
            if self.use_qk_norm:
                k = self.k_norm(k)
            if position_emb is not None:
                k = position_emb(k, start_pos=embedding_start, positions=embedding_pos)

        if use_cache:
            k, v = past_key_value.update(k, v, dim=2)
            current_key_value = past_key_value

        # Triton GQA Prefill
        if can_use_triton and HAS_TRITON_GQA and attention_mask is None:
            try:
                out = triton_gqa_forward(q, k, v)
                out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
                out = self.o_proj(out)
                if self.use_gate:
                    out = out * G
                return AttentionOutput(output=out, past_key_value=current_key_value)
            except Exception:
                pass

        # ---- 3. GQA expansion (native fallback) ----
        if self.num_kv_queries > 1:
            kv_seq_len_total = k.shape[2]
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)

        # ---- 4. Scaled dot-product attention ----
        attn_output = apply_attention(
            q, k, v,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            is_causal=self.is_causal,
            dropout=self.dropout if self.training else 0.0
        )

        output = attn_output.output
        attention_weights = attn_output.attention_weights
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        output = self.o_proj(output)

        if self.use_gate:
            output = output * G

        return AttentionOutput(
            output=output,
            attention_weights=attention_weights,
            past_key_value=current_key_value
        )
