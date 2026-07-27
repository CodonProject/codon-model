'''
From Huawei 'Caracal: Causal Architecture via Spectral Mixing' [arXiv:2605.00292 cs.LG]
'''
from codon import *
from codon.ops import AttentionOutput, apply_fourier_mixing
from codon.model.cache import BasicLayerCache


class MultiHeadFourier(BasicModel):
    '''
    Multi-Head Fourier (MHF) module.
    From 'Caracal: Causal Architecture via Spectral Mixing' [arXiv:2605.00292 cs.LG]
    Replaces dense Attention with O(L log L) frequency-domain mixing.
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        
        self.d_model = hidden_size
        self.n_heads = num_heads
        self.d_head = hidden_size // num_heads
        
        self.pre_conv = nn.Conv1d(
            in_channels=self.d_model, 
            out_channels=self.d_model,
            kernel_size=3, 
            groups=self.d_model, 
            bias=False
        )
        self.ln = nn.LayerNorm(self.d_model)
        self.silu = nn.SiLU()
        
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.W_G1 = nn.Linear(self.d_model, self.d_model)
        self.W_G2 = nn.Conv1d(
            in_channels=self.d_model, 
            out_channels=self.d_model,
            kernel_size=1, 
            groups=self.n_heads
        )
        
        self.linear = nn.Linear(self.d_model, self.d_model)
        self.linear.NEED_SCALE_INIT = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[BasicLayerCache] = None,
        **kwargs
    ) -> AttentionOutput:
        
        batch_size, seq_len, _ = hidden_states.size()
        x_permuted = hidden_states.permute(0, 2, 1)  # [B, D, L]
        
        # Determine caching behavior
        use_cache = isinstance(past_key_value, BasicLayerCache)
        is_cache_populated = use_cache and past_key_value.v_cache is not None
        
        # Conv state processing
        if is_cache_populated:
            conv_state = past_key_value.conv_state
            x_padded = torch.cat([conv_state, x_permuted], dim=2)
            x = self.pre_conv(x_padded).permute(0, 2, 1)  # [B, 1, D]
            new_conv_state = x_padded[:, :, 1:]
        else:
            padding = self.pre_conv.kernel_size[0] - 1
            x_padded = F.pad(x_permuted, (padding, 0))
            x = self.pre_conv(x_padded).permute(0, 2, 1)  # [B, L, D]
            new_conv_state = x_padded[:, :, -padding:] if use_cache else None
        x_norm = self.ln(x)
        
        x_v = self.W_V(x_norm).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        x_g = self.W_G1(x_norm).transpose(1, 2)
        x_g = self.W_G2(self.silu(x_g)).transpose(1, 2)
        x_g = x_g.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Mixing & Cache update (FourierLayerCache)
        if is_cache_populated:
            _, new_v_cache, new_g_cache = past_key_value.update(
                new_conv_state, x_v, x_g
            )
            g_flipped = torch.flip(new_g_cache, dims=[2])
            x_mixed = torch.sum(new_v_cache * g_flipped, dim=2, keepdim=True)
        else:
            x_mixed = apply_fourier_mixing(x_v, x_g, seq_len)
            if use_cache:
                past_key_value.update(new_conv_state, x_v, x_g)
        x_mixed = x_mixed.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        output = self.linear(x_mixed)
        
        current_key_value = past_key_value if use_cache else None
        
        return AttentionOutput(
            output=output,
            attention_weights=None,
            past_key_value=current_key_value
        )