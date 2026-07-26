from codon import *
from codon.block.embedding import BasicEmbedding
from codon.ops import (
    AttentionOutput,
    apply_attention
)
from codon.model.cache import BasicLayerCache


class MultiHeadAttention(BasicModel):
    ''' 
    Multi-Head Attention module.
    Supports Grouped Query Attention (GQA), QK Normalization, Gating mechanisms, 
    and Multi-Head Latent Attention (MLA).
    Attributes:
        q_proj (nn.Linear): Linear layer for query projection (Standard MHA).
        q_a_proj, q_b_proj (nn.Linear): Linear layers for query latent projection (MLA).
        k_proj, v_proj (nn.Linear): Linear layers for key/value projection (Standard MHA).
        kv_a_proj, kv_b_proj, k_p_proj, v_proj (nn.Linear): Linear layers for KV latent projection (MLA).
        o_proj (nn.Linear): Linear layer for output projection.
        q_norm (nn.RMSNorm, optional): Normalization layer for queries.
        k_norm (nn.RMSNorm, optional): Normalization layer for keys.
        q_a_norm (nn.RMSNorm, optional): Normalization layer for query latent states (MLA).
        kv_a_norm (nn.RMSNorm, optional): Normalization layer for KV latent states (MLA).
        g_proj (nn.Linear, optional): Linear layer for gating mechanism.
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
        q_lora_rank: Optional[int] = 0,
        kv_lora_rank: Optional[int] = 0,
        rope_dim: Optional[int] = 0
    ):
        '''
        Initialize the Multi-Head Attention module.
        Args:
            hidden_size (int): Size of the hidden layer.
            num_heads (int): Number of attention heads.
            num_kv_heads (int, optional): Number of key/value heads for GQA. 
                                          If None, defaults to num_heads.
            use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. 
                                          Defaults to True.
            use_gate (bool, optional): Whether to apply a gating mechanism. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
            is_causal (bool, optional): Whether to apply a causal mask. 
                                        Defaults to True (for Decoder architectures).
            q_lora_rank (int, optional): Rank for Query latent compression. If > 0, enables MLA for Q. Defaults to 0.
            kv_lora_rank (int, optional): Rank for KV latent compression. If > 0, enables MLA for KV. Defaults to 0.
            rope_dim (int, optional): Dimension of the RoPE part. Must be > 0 if MLA is enabled. Defaults to 0.
        '''
        super(MultiHeadAttention, self).__init__()
        if num_kv_heads is None: num_kv_heads = num_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.num_heads  = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim  = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate
        self.dropout = dropout
        self.is_causal = is_causal
        
        # MLA specific attributes
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim
        
        # Infer MLA usage from ranks
        self.use_mla = (self.q_lora_rank > 0) or (self.kv_lora_rank > 0)
        
        if self.use_mla:
            assert self.rope_dim > 0, 'rope_dim must be > 0 when using MLA'
            assert self.rope_dim < self.head_dim, 'rope_dim must be < head_dim'
        # Q proj
        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_norm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, hidden_size, bias=bias)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # KV proj
        if self.kv_lora_rank > 0:
            self.kv_a_proj = nn.Linear(hidden_size, self.kv_lora_rank, bias=bias)
            self.kv_a_norm = nn.RMSNorm(self.kv_lora_rank)
            self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.head_dim - self.rope_dim), bias=bias)
            self.v_proj = nn.Linear(self.kv_lora_rank, hidden_size, bias=bias)
            self.k_p_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.rope_dim, bias=bias)
        else:
            self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
            self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        # Norms
        if use_qk_norm:
            norm_dim = self.head_dim - self.rope_dim if (self.use_mla and self.rope_dim > 0) else self.head_dim
            self.q_norm = nn.RMSNorm(norm_dim)
            self.k_norm = nn.RMSNorm(norm_dim)
        
        # Gate
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        # Output
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
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
        Perform forward pass of Multi-Head Attention.
        Args:
            hidden_states (torch.Tensor): Input hidden states.
            kv_states (torch.Tensor, optional): Hidden states for keys/values. 
                                                If None, uses hidden_states. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. 
                                                     Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. 
                                                Defaults to False.
            position_emb (BasicEmbedding, optional): Positional embedding module. 
                                                    Defaults to None.
            embedding_start (int, optional): Starting position for embedding. Defaults to 0.
            embedding_pos (torch.Tensor, optional): Explicit position indices for positional embedding. 
                                                    Defaults to None.
        
        Returns:
            AttentionOutput: Object containing output, attention weights, and KV cache.
        '''
        
        if kv_states is None:
            kv_states = hidden_states
        
        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]
        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        # Determine if cache is used by verifying the class instance
        use_cache = isinstance(past_key_value, BasicLayerCache)

        # Query Processing
        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.rope_dim > 0:
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
        
        # Key & Value Processing
        current_key_value = None
        
        if self.kv_lora_rank > 0:
            # MLA KV Path (Expects TensorLayerCache with concat_dim=1)
            kv_latent = self.kv_a_norm(self.kv_a_proj(kv_states))
            
            past_len = past_key_value.seq_length if use_cache else 0
            if use_cache:
                kv_latent = past_key_value.update(kv_latent)
                current_key_value = past_key_value
                
            kv_len_total = kv_latent.shape[1]
            
            k_c = self.kv_b_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.head_dim - self.rope_dim).transpose(1, 2)
            v = self.v_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.head_dim).transpose(1, 2)
            k_p = self.k_p_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.rope_dim).transpose(1, 2)
            
            if self.use_qk_norm:
                k_c = self.k_norm(k_c)
                
            if position_emb is not None:
                # For cached MLA, kv_latent contains the full history, so positions start from 0 if cache is already populated
                if use_cache and past_len > 0:
                    k_p = position_emb(k_p, start_pos=0, positions=None)
                else:
                    k_p = position_emb(k_p, start_pos=embedding_start, positions=embedding_pos)
                    
            k = torch.cat([k_c, k_p], dim=-1)
            
        else:
            # Standard KV Path (Expects KVLayerCache)
            k = self.k_proj(kv_states)
            v = self.v_proj(kv_states)
            k = k.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
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
        
        # GQA Expansion
        if self.num_kv_queries > 1 and self.kv_lora_rank == 0:
            kv_seq_len_total = k.shape[2]
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            
        # Attention Computation
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



class MultiHeadAttentionKEV(BasicModel):
    ''' 
    Multi-Head Attention module where K = V (Key and Value are identical, Q is independent).
    Supports Grouped Query Attention (GQA), QK Normalization, and Gating mechanisms.

    Attributes:
        q_proj (nn.Linear): Linear layer for query projection.
        kv_proj (nn.Linear): Linear layer for key-value projection.
        o_proj (nn.Linear): Linear layer for output projection.
        q_norm (nn.RMSNorm, optional): Normalization layer for queries.
        k_norm (nn.RMSNorm, optional): Normalization layer for keys (and values since K = V).
        g_proj (nn.Linear, optional): Linear layer for gating mechanism.
    '''
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads=None,
        use_qk_norm=True,
        use_gate=False,
        dropout=0.1,
        bias: bool=True,
        is_causal=True
    ):
        '''
        Initialize the Multi-Head Attention module where K = V.
        From 'Do Transformers Need Three Projections? Systematic Study of QKV Variants' [arXiv:2606.04032 cs.LG]

        Args:
            hidden_size (int): Size of the hidden layer.
            num_heads (int): Number of attention heads.
            num_kv_heads (int, optional): Number of key/value heads for GQA. 
                                          If None, defaults to num_heads.
            use_qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. 
                                          Defaults to True.
            use_gate (bool, optional): Whether to apply a gating mechanism. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            is_causal (bool, optional): Whether to apply a causal mask. 
                                        Defaults to True (for Decoder architectures).
        '''
        super(MultiHeadAttentionKEV, self).__init__()

        if num_kv_heads is None: num_kv_heads = num_heads

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_heads  = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim  = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate
        self.dropout = dropout
        self.is_causal = is_causal
        
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
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
        optimize_kv_cache: bool = True
    ) -> AttentionOutput:
        ''' 
        Perform forward pass of Multi-Head Attention where K = V.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            kv_states (torch.Tensor, optional): Hidden states for keys/values. 
                                                If None, uses hidden_states. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. 
                                                     Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. 
                                                Defaults to False.
            position_emb (BasicEmbedding, optional): Positional embedding module. 
                                                    Defaults to None.
            embedding_start (int, optional): Starting position for embedding. Defaults to 0.
            embedding_pos (torch.Tensor, optional): Explicit position indices for positional embedding. 
                                                    Defaults to None.
            optimize_kv_cache (bool, optional): Whether to optimize KV cache to store a single tensor.
                                                 Defaults to True.
        
        Returns:
            AttentionOutput: Object containing output, attention weights, and KV cache.
        '''
        
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        use_cache = isinstance(past_key_value, BasicLayerCache)

        Q = self.q_proj(hidden_states)
        KV = self.kv_proj(kv_states)

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        KV = KV.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if optimize_kv_cache:
            # Expects TensorLayerCache with concat_dim=2
            past_len = past_key_value.seq_length if use_cache else 0
            if use_cache:
                KV = past_key_value.update(KV)
                current_key_value = past_key_value
            else:
                current_key_value = None

            K = KV
            V = KV

            if self.use_qk_norm:
                Q = self.q_norm(Q)
                K = self.k_norm(K)
            
            if position_emb is not None:
                Q = position_emb(Q, start_pos=embedding_start, positions=embedding_pos)
                
                k_start = embedding_start - past_len
                if embedding_pos is not None:
                    offset = embedding_start - past_len
                    if embedding_pos.ndim == 2:
                        past_pos = (torch.arange(past_len, device=embedding_pos.device) + offset).unsqueeze(0).expand(batch_size, -1)
                        full_pos = torch.cat((past_pos, embedding_pos), dim=1)
                    elif embedding_pos.ndim == 3:
                        num_axes = embedding_pos.shape[-1]
                        past_pos = (torch.arange(past_len, device=embedding_pos.device) + offset).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, num_axes)
                        full_pos = torch.cat((past_pos, embedding_pos), dim=1)
                    else:
                        full_pos = embedding_pos
                    K = position_emb(K, start_pos=k_start, positions=full_pos)
                else:
                    K = position_emb(K, start_pos=k_start, positions=None)
        else:
            # Expects KVLayerCache
            K = KV
            V = KV

            if self.use_qk_norm:
                Q = self.q_norm(Q)
                K = self.k_norm(K)
            
            if position_emb is not None:
                Q = position_emb(Q, start_pos=embedding_start, positions=embedding_pos)
                K = position_emb(K, start_pos=embedding_start, positions=embedding_pos)
            
            if use_cache:
                K, V = past_key_value.update(K, V, dim=2)
                current_key_value = past_key_value
            else:
                current_key_value = None

        kv_seq_len_total = K.shape[2]

        if self.num_kv_queries > 1:
            # [B, H_kv, 1, L, D] -> [B, H_kv, G, L, D]
            K = K[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            V = V[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            
            K = K.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            V = V.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            
        attn_output = apply_attention(
            Q, K, V, 
            attention_mask=attention_mask, 
            output_attentions=output_attentions,
            is_causal=self.is_causal,
            dropout=self.dropout if self.training else 0.0
        )

        output = attn_output.output
        attention_weights = attn_output.attention_weights
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        output = self.o_proj(output)

        if self.use_gate: output = output * G

        return AttentionOutput(
            output=output,
            attention_weights=attention_weights,
            past_key_value=current_key_value
        )


