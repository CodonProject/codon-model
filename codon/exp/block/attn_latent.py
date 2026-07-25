from codon import *
from codon.block.embedding import BasicEmbedding
from codon.ops import (
    AttentionOutput,
    apply_attention
)


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
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int]=None,
        use_qk_norm: Optional[bool]=True,
        use_gate: Optional[bool]=False,
        dropout: Optional[float]=0.1,
        bias: Optional[bool]=True,
        is_causal: Optional[bool]=True,
        q_lora_rank: Optional[int]=0,
        kv_lora_rank: Optional[int]=0,
        rope_dim: Optional[int]=0
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
        past_key_value: tuple = None,
        use_cache: bool = False
    ) -> AttentionOutput:
        ''' 
        Perform forward pass of Multi-Head Attention.
        '''
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
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
            # MLA KV Path
            kv_latent = self.kv_a_norm(self.kv_a_proj(kv_states))
            
            if use_cache:
                if past_key_value is not None:
                    past_kv = past_key_value[0]
                    kv_latent = torch.cat((past_kv, kv_latent), dim=1)
                current_key_value = (kv_latent,)
                
            kv_len_total = kv_latent.shape[1]
            
            k_c = self.kv_b_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.head_dim - self.rope_dim).transpose(1, 2)
            v = self.v_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.head_dim).transpose(1, 2)
            k_p = self.k_p_proj(kv_latent).view(batch_size, kv_len_total, self.num_heads, self.rope_dim).transpose(1, 2)
            
            if self.use_qk_norm:
                k_c = self.k_norm(k_c)
                
            if position_emb is not None:
                # For cached MLA, kv_latent contains the full history, so positions start from 0
                if past_key_value is not None:
                    k_p = position_emb(k_p, start_pos=0, positions=None)
                else:
                    k_p = position_emb(k_p, start_pos=embedding_start, positions=embedding_pos)
                    
            k = torch.cat([k_c, k_p], dim=-1)
            
        else:
            # Standard KV Path
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
                if past_key_value is not None:
                    past_k, past_v = past_key_value
                    k = torch.cat((past_k, k), dim=2)
                    v = torch.cat((past_v, v), dim=2)
                current_key_value = (k, v)

        # GQA Expansion
        # Only expand if using standard KV path and num_kv_heads < num_heads
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