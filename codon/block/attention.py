from codon import *
from codon.block.embedding import BasicEmbedding
from codon.ops import (
    AttentionOutput,
    apply_attention
)
from codon.ops.attn_cuda import *
from codon.model.cache import (
    BasicLayerCache,
    KVLayerCache,
    TensorLayerCache,
    HCALayerCache,
    CSALayerCache
)


class MultiHeadAttention(BasicModel):
    ''' 
    Multi-Head Attention module.
    Supports:
        - Standard MHA & Grouped Query Attention (GQA)
        - QK Normalization & Gating mechanism
        - Multi-Head Latent Attention  (MLA)
        - Heavily Compressed Attention (HCA) [Low-rank + Block Merge + FP4 Quantization]
        - Compressed Sparse Attention  (CSA) [Block Compress + Lightning Indexer + Top-K Sparsity]
        - Triton GPU Accelerated Kernels (Auto-enabled for Prefill q_len > 1)
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
        # MLA Config
        q_lora_rank: Optional[int] = 0,
        kv_lora_rank: Optional[int] = 0,
        rope_dim: Optional[int] = 0,
        # HCA Config
        use_hca: Optional[bool] = False,
        hca_latent_dim: Optional[int] = 512,
        hca_block_size: Optional[int] = 128,
        hca_fp4_storage: Optional[bool] = True,
        # CSA Config
        use_csa: Optional[bool] = False,
        csa_compressed_dim: Optional[int] = 512,
        csa_block_size: Optional[int] = 4,
        csa_top_k: Optional[int] = 128,
        # Triton ops switch
        use_triton: bool = False,
    ):
        super(MultiHeadAttention, self).__init__()
        if num_kv_heads is None: num_kv_heads = num_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        assert not (use_hca and use_csa), "Cannot enable both HCA and CSA simultaneously."
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_queries = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_qk_norm = use_qk_norm
        self.use_gate = use_gate
        self.dropout = dropout
        self.is_causal = is_causal
        
        self.use_hca = use_hca
        self.use_csa = use_csa
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim

        self.use_mla = (self.q_lora_rank > 0) or (self.kv_lora_rank > 0)
        self.use_triton = use_triton

        if self.use_mla:
            assert self.rope_dim > 0, 'rope_dim must be > 0 when using MLA'
            assert self.rope_dim < self.head_dim, 'rope_dim must be < head_dim'

        # 1. Query Projections
        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_norm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, hidden_size, bias=bias)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # 2. Key/Value Projections
        if self.use_hca:
            self.hca_latent_dim = hca_latent_dim
            self.hca_block_size = hca_block_size
            self.hca_fp4_storage = hca_fp4_storage
            
            self.hca_kv_proj = nn.Linear(hidden_size, hca_latent_dim, bias=False)
            self.hca_k_decompress = nn.Linear(hca_latent_dim, hidden_size, bias=False)
            self.hca_v_decompress = nn.Linear(hca_latent_dim, hidden_size, bias=False)
            self.hca_block_weights = nn.Parameter(
                torch.randn(hca_block_size, hca_latent_dim) / (hca_latent_dim ** 0.5)
            )
        elif self.use_csa:
            self.csa_compressed_dim = csa_compressed_dim
            self.csa_block_size = csa_block_size
            self.csa_top_k = csa_top_k
            
            self.csa_kv_compress = nn.Linear(hidden_size, csa_compressed_dim, bias=False)
            self.csa_q_compress = nn.Linear(hidden_size, csa_compressed_dim, bias=False)
            
            self.csa_indexer = nn.Sequential(
                nn.Linear(csa_compressed_dim, csa_compressed_dim // 2),
                nn.GELU(),
                nn.Linear(csa_compressed_dim // 2, 1)
            )
            self.k_proj = nn.Linear(csa_compressed_dim, self.kv_dim, bias=bias)
            self.v_proj = nn.Linear(csa_compressed_dim, self.kv_dim, bias=bias)
        elif self.kv_lora_rank > 0:
            self.kv_a_proj = nn.Linear(hidden_size, self.kv_lora_rank, bias=bias)
            self.kv_a_norm = nn.RMSNorm(self.kv_lora_rank)
            self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.head_dim - self.rope_dim), bias=bias)
            self.v_proj = nn.Linear(self.kv_lora_rank, hidden_size, bias=bias)
            self.k_p_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.rope_dim, bias=bias)
        else:
            self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
            self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)

        if use_qk_norm:
            norm_dim = self.head_dim - self.rope_dim if (self.use_mla and self.rope_dim > 0) else self.head_dim
            self.q_norm = nn.RMSNorm(norm_dim)
            self.k_norm = nn.RMSNorm(norm_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    # HCA tool
    def _hca_compress_to_fp4(self, tensor: torch.Tensor):
        if not self.hca_fp4_storage:
            return tensor
            
        min_val = tensor.min(dim=-1, keepdim=True).values
        max_val = tensor.max(dim=-1, keepdim=True).values
        scale = (max_val - min_val) / 15.0
        scale_clamp = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        quantized = ((tensor - min_val) / scale_clamp).round().clamp(0, 15).to(torch.uint8)
        
        if quantized.shape[1] % 2 != 0:
            quantized = F.pad(quantized, (0, 0, 0, 1))
            min_val = F.pad(min_val, (0, 0, 0, 1))
            scale_clamp = F.pad(scale_clamp, (0, 0, 0, 1))
        high = quantized[:, 1::2, :]
        low = quantized[:, 0::2, :]
        packed = low | (high << 4)
        
        return packed, min_val, scale_clamp
    
    def _hca_decompress_from_fp4(self, packed: torch.Tensor, min_val: torch.Tensor, scale: torch.Tensor):
        if not self.hca_fp4_storage:
            return packed
            
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        
        B, num_packed_blocks, D = packed.shape
        unpacked = torch.empty(B, num_packed_blocks * 2, D, dtype=torch.uint8, device=packed.device)
        unpacked[:, 0::2, :] = low
        unpacked[:, 1::2, :] = high
        
        unpacked = unpacked[:, :min_val.shape[1], :]
        
        return unpacked.float() * scale + min_val
    
    def _hca_block_merge(self, kv_latent: torch.Tensor) -> torch.Tensor:
        B, L, D = kv_latent.shape
        pad_len = (self.hca_block_size - (L % self.hca_block_size)) % self.hca_block_size
        if pad_len > 0:
            kv_latent = F.pad(kv_latent, (0, 0, 0, pad_len))
            L = kv_latent.shape[1]
            
        num_blocks = L // self.hca_block_size
        kv_blocks = kv_latent.view(B, num_blocks, self.hca_block_size, D)
        weights = F.softmax(self.hca_block_weights, dim=0).unsqueeze(0).unsqueeze(0)
        merged = torch.sum(kv_blocks * weights, dim=2)
        return merged
    
    # CSA tool
    def _csa_block_compress(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        pad_len = (self.csa_block_size - (L % self.csa_block_size)) % self.csa_block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            L = x.shape[1]
        num_blocks = L // self.csa_block_size
        x_blocks = x.view(B, num_blocks, self.csa_block_size, D)
        return x_blocks.mean(dim=2)
    
    def _csa_lightning_indexer(self, q_blocks: torch.Tensor, kv_blocks: torch.Tensor) -> torch.Tensor:
        raw_scores = torch.bmm(q_blocks, kv_blocks.transpose(1, 2)) / (self.csa_compressed_dim ** 0.5)
        q_score = self.csa_indexer(q_blocks)
        k_score = self.csa_indexer(kv_blocks).transpose(1, 2)
        refined_scores = q_score + k_score
        return raw_scores * 0.7 + refined_scores * 0.3
    
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
        
        if kv_states is None:
            kv_states = hidden_states
        
        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]
        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        # 仅在 Prefill 阶段 (q_len > 1) 且显卡支持时启用 Triton Block 算子
        can_use_triton = self.use_triton and hidden_states.is_cuda and q_len > 1 and not output_attentions

        # 1. Query Processing
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

        # 2. Key & Value Processing Based on Mode
        current_key_value = None
        if self.use_hca:
            use_cache = isinstance(past_key_value, HCALayerCache)
            kv_latent = self.hca_kv_proj(kv_states)
            merged_kv = self._hca_block_merge(kv_latent)
            if use_cache:
                if self.hca_fp4_storage:
                    quantized, min_val, scale = self._hca_compress_to_fp4(merged_kv)
                    q_c, m_v, sc = past_key_value.update_fp4(quantized, min_val, scale, merged_kv.shape[1])
                    merged_kv = self._hca_decompress_from_fp4(q_c, m_v, sc)
                else:
                    merged_kv = past_key_value.update_raw(merged_kv)
                current_key_value = past_key_value
            
            # Triton HCA Prefill
            if can_use_triton and HAS_TRITON_HCA:
                try:
                    out = triton_hca_forward(
                        q, merged_kv,
                        self.hca_k_decompress.weight.t(),
                        self.hca_v_decompress.weight.t()
                    )
                    out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
                    out = self.o_proj(out)
                    if self.use_gate: out = out * G
                    return AttentionOutput(output=out, past_key_value=current_key_value)
                except Exception: pass

            num_blocks = merged_kv.shape[1]
            K_res = self.hca_k_decompress(merged_kv)
            V_res = self.hca_v_decompress(merged_kv)
            k = K_res.view(batch_size, num_blocks, self.num_heads, self.head_dim).transpose(1, 2)
            v = V_res.view(batch_size, num_blocks, self.num_heads, self.head_dim).transpose(1, 2)
            if self.use_qk_norm:
                k = self.k_norm(k)

        elif self.use_csa:
            use_cache = isinstance(past_key_value, CSALayerCache)
            kv_compressed = self.csa_kv_compress(kv_states)
            kv_blocks = self._csa_block_compress(kv_compressed)
            if use_cache:
                kv_blocks = past_key_value.update(kv_blocks)
                current_key_value = past_key_value
            q_compressed = self.csa_q_compress(hidden_states)
            q_blocks = self._csa_block_compress(q_compressed)
            block_scores = self._csa_lightning_indexer(q_blocks, kv_blocks)
            global_scores = block_scores.mean(dim=1)
            
            top_k = min(self.csa_top_k, kv_blocks.shape[1])
            topk_values, topk_indices = torch.topk(global_scores, k=top_k, dim=-1)
            selected_kv = torch.stack([
                torch.index_select(kv_blocks[b], 0, topk_indices[b]) for b in range(batch_size)
            ], dim=0)

            # Triton CSA Prefill
            if can_use_triton and HAS_TRITON_CSA:
                try:
                    out = triton_csa_forward(
                        q, selected_kv,
                        self.k_proj.weight.t(),
                        self.v_proj.weight.t()
                    )
                    out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
                    out = self.o_proj(out)
                    if self.use_gate: out = out * G
                    return AttentionOutput(output=out, past_key_value=current_key_value)
                except Exception: pass

            k = self.k_proj(selected_kv).view(batch_size, top_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(selected_kv).view(batch_size, top_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if self.use_qk_norm:
                k = self.k_norm(k)

        elif self.kv_lora_rank > 0:
            # MLA
            use_cache = isinstance(past_key_value, TensorLayerCache)
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
                if use_cache and past_len > 0:
                    k_p = position_emb(k_p, start_pos=0, positions=None)
                else:
                    k_p = position_emb(k_p, start_pos=embedding_start, positions=embedding_pos)
            
            # Triton MLA Prefill
            if can_use_triton and HAS_TRITON_MLA:
                try:
                    q_c, q_p = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
                    w_kc = self.kv_b_proj.weight.view(self.num_heads, self.head_dim - self.rope_dim, self.kv_lora_rank)
                    w_vc = self.v_proj.weight.view(self.num_heads, self.kv_lora_rank, self.head_dim)
                    out = triton_mla_forward(q_c, q_p, kv_latent, k_p, w_kc, w_vc)
                    out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
                    out = self.o_proj(out)
                    if self.use_gate: out = out * G
                    return AttentionOutput(output=out, past_key_value=current_key_value)
                except Exception: pass

            k = torch.cat([k_c, k_p], dim=-1)
        else:
            # MHA / GQA
            use_cache = isinstance(past_key_value, KVLayerCache)
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
                
            # Triton GQA Prefill
            if can_use_triton and HAS_TRITON_GQA and attention_mask is None:
                try:
                    out = triton_gqa_forward(q, k, v)
                    out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
                    out = self.o_proj(out)
                    if self.use_gate: out = out * G
                    return AttentionOutput(output=out, past_key_value=current_key_value)
                except Exception: pass

        # 3. GQA Expansion (Native Fallback)
        if self.num_kv_queries > 1 and not (self.kv_lora_rank > 0 or self.use_hca):
            kv_seq_len_total = k.shape[2]
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_queries, kv_seq_len_total, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, kv_seq_len_total, self.head_dim)
        
        # 4. Native PyTorch Attention Computation (Optimized for Decode GEMV)
        attn_output = apply_attention(
            q, k, v,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            is_causal=self.is_causal if not (self.use_hca or self.use_csa) else False,
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


