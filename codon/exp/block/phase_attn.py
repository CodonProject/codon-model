from codon import *
from codon.block.complex import ComplexLinear


class PhaseCrossAttention(BasicModel):
    def __init__(
        self,
        num_features: int,
        tau_attn: float = 1.0,
        use_softmax: bool = True,
        disable_proj: bool = False,
        disable_residual: bool = False,
        pure_phase_gate: bool = True,
        track: bool = True,
        mode: Literal['element', 'matrix'] = 'element',
        num_heads: int = 4
    ):
        super().__init__()

        mode = mode.lower().strip()
        if mode not in ['element', 'matrix']:
            raise ValueError()

        self.num_features = num_features
        self.tau_attn = tau_attn
        self.use_softmax = use_softmax
        self.disable_proj = disable_proj
        self.disable_residual = disable_residual
        self.pure_phase_gate = pure_phase_gate
        self.track = track
        self.mode = mode

        self.num_heads = num_heads
        self.head_dim = num_features // num_heads

        self.q_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()
        self.k_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()
        self.v_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()
        self.o_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj and mode == 'matrix' else nn.Identity()

        self.alpha = nn.Parameter(torch.tensor(0.5)) if not disable_residual else None

        self.last_cos_phase_diff = None

    def forward(
        self,
        x_original: torch.Tensor,
        x_l23: torch.Tensor,
        x_higher: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        Q: torch.Tensor = self.q_proj(x_higher)
        K: torch.Tensor = self.k_proj(x_l23)
        V: torch.Tensor = self.v_proj(x_original)
        
        if self.track:
            Q_mag = torch.abs(Q)
            K_mag = torch.abs(K)
            valid = (Q_mag > 1e-6) & (K_mag > 1e-6)
            if valid.any():
                cos_phase_diff = torch.cos(torch.angle(Q) - torch.angle(K))
                self.last_cos_phase_diff = cos_phase_diff[valid].mean().item()
            else:
                self.last_cos_phase_diff = float('nan')

        if self.mode == 'matrix':
            Q = Q.view(-1, self.num_heads, self.head_dim)
            K = K.view(-1, self.num_heads, self.head_dim)
            V = V.view(-1, self.num_heads, self.head_dim)

            # [B, num_heads, head_dim] @ [B, num_heads, head_dim].H → [B, num_heads, num_heads]
            scores = torch.einsum('bhd,bgd->bhg', Q, K.conj()).real
            attn_weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
            
            out = torch.einsum('bhg,bgd->bhd', attn_weights.to(V.dtype), V)
            out = out.reshape(-1, self.num_features)

            out = self.o_proj(out)
            
            if not self.disable_residual:
                out = out + x_l23 * self.alpha

            return out, attn_weights
        

        complex_interfere = Q * K.conj()
        if self.pure_phase_gate:
            cos_phi = complex_interfere.real / (complex_interfere.abs() + 1e-6)
            scores = cos_phi / self.tau_attn
        else:
            scores = complex_interfere.real / self.tau_attn
        
        if self.use_softmax:
            attn_weights = torch.softmax(scores / (self.num_features ** 0.5), dim=-1)
        else:
            attn_weights = torch.sigmoid(scores)

        attn_weights_complex = torch.complex(attn_weights, torch.zeros_like(attn_weights))

        gated_v = attn_weights_complex * V

        if self.disable_residual:
            z_l5_output = gated_v
        else:
            z_l5_output = gated_v + x_l23 * self.alpha

        return z_l5_output, attn_weights
