from codon.base  import *
from codon.block.embedding import BaseEmbedding
from codon.ops.attention   import AttentionOutput, apply_attention


class MultiHeadAttention(BasicModel):
    ''' 
    Multi-Head Attention module.
    Supports Grouped Query Attention (GQA), QK Normalization, and Gating mechanisms.

    Attributes:
        q_proj (nn.Linear): Linear layer for query projection.
        k_proj (nn.Linear): Linear layer for key projection.
        v_proj (nn.Linear): Linear layer for value projection.
        o_proj (nn.Linear): Linear layer for output projection.
        q_norm (nn.RMSNorm, optional): Normalization layer for queries.
        k_norm (nn.RMSNorm, optional): Normalization layer for keys.
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
        is_causal=True
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
            is_causal (bool, optional): Whether to apply a causal mask. 
                                        Defaults to True (for Decoder architectures).
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
        
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        output_attentions: bool = False,
        position_emb: BaseEmbedding = None,
        embedding_pos: int = 0,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> AttentionOutput:
        ''' 
        Perform forward pass of Multi-Head Attention.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            kv_states (torch.Tensor, optional): Hidden states for keys/values. 
                                                If None, uses hidden_states. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask (usually Padding Mask). 
                                                     Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. 
                                                Defaults to False.
            position_emb (BaseEmbedding, optional): Positional embedding module. 
                                                    Defaults to None.
            embedding_pos (int, optional): Starting position for embedding. Defaults to 0.
            past_key_value (tuple[torch.Tensor, torch.Tensor], optional): Past key-value cache. 
                                                                          Defaults to None.
            use_cache (bool, optional): Whether to use KV cache. Defaults to False.
        
        Returns:
            AttentionOutput: Object containing output, attention weights, and KV cache.
        '''
        
        if kv_states is None:
            kv_states = hidden_states

        batch_size, q_len, _ = hidden_states.shape
        kv_len_input = kv_states.shape[1]

        if self.use_gate:
            G = torch.sigmoid(self.g_proj(hidden_states))
        
        Q = self.q_proj(hidden_states)
        K = self.k_proj(kv_states)
        V = self.v_proj(kv_states)

        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len_input, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        if position_emb is not None:
            Q = position_emb(Q, start_pos=embedding_pos)
            K = position_emb(K, start_pos=embedding_pos)
        
        current_key_value = None
        if use_cache:
            if past_key_value is not None:
                past_k, past_v = past_key_value
                K = torch.cat((past_k, K), dim=2)
                V = torch.cat((past_v, V), dim=2)
            current_key_value = (K, V)

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
