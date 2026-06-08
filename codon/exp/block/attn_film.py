from codon.base  import *
from codon.block import BasicEmbedding, FiLM
from codon.ops.attention import AttentionOutput, apply_attention


class MultiHeadAttentionFiLM(BasicModel):
    ''' 
    Multi-Head Attention with FiLM-modulated Key and Value states.
    K and V are dynamically modulated by a dimension-reduced Query before head splitting.
    Supports Grouped Query Attention (GQA), QK Normalization, and Gating.
    '''
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads=None,
        cond_dim=None,
        use_qk_norm=True,
        use_gate=False,
        dropout=0.1,
        bias: bool = True,
        is_causal=True,
        # FiLM parameters
        use_beta=True,
        use_gamma=True,
        use_film_gate=False
    ):
        super(MultiHeadAttentionFiLM, self).__init__()

        if num_kv_heads is None: 
            num_kv_heads = num_heads

        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0

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
        
        # FiLM Condition dimension (defaults to 1/4 of hidden_size)
        self.cond_dim = cond_dim if cond_dim is not None else hidden_size // 4

        # Linear layer to down-project Q to condition space
        self.q_down_proj = nn.Linear(hidden_size, self.cond_dim, bias=bias)

        # FiLM modulators for K and V
        self.film_k = FiLM(
            in_features=self.kv_dim,
            cond_features=self.cond_dim,
            use_beta=use_beta,
            use_gamma=use_gamma,
            use_gate=use_film_gate
        )
        self.film_v = FiLM(
            in_features=self.kv_dim,
            cond_features=self.cond_dim,
            use_beta=use_beta,
            use_gamma=use_gamma,
            use_gate=use_film_gate
        )

        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
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
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> AttentionOutput:
        
        if kv_states is not None and kv_states is not hidden_states:
            raise ValueError(
                "MultiHeadAttentionFiLM only supports self-attention "
                "where query and key/value source sequences are identical."
            )

        batch_size, q_len, _ = hidden_states.shape

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        Q = self.q_proj(hidden_states)   # [B, L, H]
        K = self.k_proj(hidden_states)   # [B, L, KV_dim]
        V = self.v_proj(hidden_states)   # [B, L, KV_dim]

        q_cond = self.q_down_proj(Q)     # [B, L, cond_dim]

        K = self.film_k(K, q_cond).final_output  # [B, L, KV_dim]
        V = self.film_v(V, q_cond).final_output  # [B, L, KV_dim]

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        if position_emb is not None:
            Q = position_emb(Q, start_pos=embedding_start, positions=embedding_pos)
            K = position_emb(K, start_pos=embedding_start, positions=embedding_pos)
        
        current_key_value = None
        if use_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                K = torch.cat((past_k, K), dim=2)
                V = torch.cat((past_v, V), dim=2)
            current_key_value = (K, V)

        kv_seq_len_total = K.shape[2]

        if self.num_kv_queries > 1:
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

        if self.use_gate: 
            output = output * G

        return AttentionOutput(
            output=output,
            attention_weights=attention_weights,
            past_key_value=current_key_value
        )