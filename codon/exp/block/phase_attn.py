from codon.base import *
from codon.block.complex import ComplexLinear


class PhaseCrossAttention(BasicModel):
    def __init__(
        self,
        num_features: int,
        tau_attn: float = 1.0,
        use_softmax: bool = True,
        disable_proj: bool = False
    ):
        super().__init__()

        self.num_features = num_features
        self.tau_attn = tau_attn
        self.use_softmax = use_softmax
        self.disable_proj = disable_proj

        self.q_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()
        self.k_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()
        self.v_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if not disable_proj else nn.Identity()

        self.last_cos_phase_diff = None

    def forward(
        self,
        x_original: torch.Tensor,
        x_l23: torch.Tensor,
        x_higher: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        Q = self.q_proj(x_higher)
        K = self.k_proj(x_l23)
        V = self.v_proj(x_original)
        
        complex_interfere = Q * K.conj()
        scores = complex_interfere.real / self.tau_attn
        if self.use_softmax:
            attn_weights = torch.softmax(scores / (self.num_features ** 0.5), dim=-1)
        else:
            attn_weights = torch.sigmoid(scores)

        attn_weights_complex = torch.complex(attn_weights, torch.zeros_like(attn_weights))

        gated_v = attn_weights_complex * V
        z_l5_output = gated_v + x_l23
        
        theta_q = torch.angle(Q)
        theta_k = torch.angle(K)
        cos_phase_diff = torch.cos(theta_q - theta_k)
        self.last_cos_phase_diff = cos_phase_diff.detach().mean().item()

        return z_l5_output, attn_weights