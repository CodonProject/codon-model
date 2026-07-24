from codon import *
from codon.block.complex import ComplexLinear, ModReLU, PhaseShift
from codon.exp.block.phase_attn import PhaseCrossAttention


@torch.no_grad()
def enforce_scaling(model: Union[nn.Module, BasicModel], scale: float, eps: float = 1e-8):
    params_dict = dict(model.named_parameters())
    processed = set()

    if isinstance(model, BasicModel): model.trigger('cortex_scaling')
    
    for name, param in params_dict.items():
        if name in processed: continue
        
        if param.is_complex():
            magnitude = torch.abs(param)
            param.copy_((param / (magnitude + eps)) * scale)
            processed.add(name)
            
        elif 'real' in name:
            imag_name = name.replace('real', 'imag')
            if imag_name in params_dict:
                real_param = param
                imag_param = params_dict[imag_name]
                
                magnitude = torch.sqrt(real_param**2 + imag_param**2)
                scale_factor = scale / (magnitude + eps)
                
                real_param.copy_(real_param * scale_factor)
                imag_param.copy_(imag_param * scale_factor)
                
                processed.add(name)
                processed.add(imag_name)


class ApicalDendriteIntegration(BasicModel):
    '''
    Layer 1
    '''
    def __init__(
        self,
        num_features: int
    ):
        super().__init__()

        self.num_features = num_features

        self.pfc = ComplexLinear(in_features=num_features, out_features=num_features)
        self.nmda_threshold = nn.Parameter(torch.tensor(0.5))
        self.nmda_gain = nn.Parameter(torch.tensor(2.0))

        self.last_nmda_boost = None
    
    def forward(
        self,
        x_pfc: Optional[torch.Tensor],
        x_thalamus_phase: Optional[torch.Tensor],
        batch_size: int = None
    ) -> torch.Tensor:
        
        if x_pfc is None:
            B = batch_size if batch_size is not None else 1
            return torch.zeros(B, self.num_features, dtype=self.dtype[0], device=self.device)
        
        x_pfc: torch.Tensor = self.pfc(x_pfc)

        amplitude = torch.abs(x_pfc)
        phase = torch.angle(x_pfc)

        safe_threshold = 2.0 * torch.sigmoid(self.nmda_threshold)
        safe_gain      = 5.0 * torch.sigmoid(self.nmda_gain)

        nmda_boost = torch.sigmoid(amplitude - safe_threshold) * safe_gain
        boosted_amplitude = amplitude + nmda_boost

        self.last_nmda_boost = nmda_boost.detach().mean().item()
        
        x_integrated = torch.polar(boosted_amplitude, phase)
        
        if x_thalamus_phase is not None:
            theta_thalamus = torch.angle(x_thalamus_phase) if x_thalamus_phase.is_complex() else x_thalamus_phase
            phase_diff = phase - theta_thalamus
            thalamic_gate = 0.5 * (torch.cos(phase_diff) + 1.0)
            x_integrated = x_integrated * thalamic_gate

        return x_integrated


class L23Integration(BasicModel):
    '''
    Layer 2/3
    '''
    def __init__(
        self,
        num_features: int,
        phase_shift: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.use_phase_shift = phase_shift

        self.phase_shift = PhaseShift(num_features=num_features) if phase_shift else None

        self.sst_base_inhibition = nn.Parameter(torch.tensor(0.8))
        self.w_vip_to_sst = nn.Parameter(torch.tensor(1.2))
        self.pv_gamma = nn.Parameter(torch.tensor(0.1))

        self.ff_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False)
        self.fb_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False)
        
        self.act = ModReLU(num_features=num_features, initial_bias=-0.05)

        self.last_g_sst = None
        self.last_pv_shunting = None
    
    def cortex_scaling(self):
        with torch.no_grad():
            for p in self.act.parameters(): p.clamp_(max=0.0)
    
    def forward(
        self,
        x_original: torch.Tensor,
        x_higher: Optional[torch.Tensor] 
    ) -> torch.Tensor:

        if self.use_phase_shift:
            x_original: torch.Tensor = self.phase_shift(x_original)
        
        if x_higher is not None:
            # active VIP↑ → SST↓
            safe_sst_base = torch.sigmoid(self.sst_base_inhibition)
            safe_w_vip    = 5.0 * torch.sigmoid(self.w_vip_to_sst)

            a_vip = torch.mean(torch.abs(x_higher), dim=-1, keepdim=True)
            g_sst = torch.clamp(safe_sst_base - safe_w_vip * a_vip, min=0.0, max=1.0)
            
            # gate SST↓ → higher↑
            x_fb_gated = x_higher * (1.0 - g_sst)
            self.last_g_sst = g_sst.detach().mean().item()
        else:
            x_fb_gated = torch.zeros_like(x_original)
            self.last_g_sst = 1.0
        
        z_ff = self.ff_proj(x_original)
        z_fb = self.fb_proj(x_fb_gated)

        z_integ = z_ff + z_fb

        a_pv = torch.mean(torch.abs(z_integ), dim=-1, keepdim=True)
        
        safe_pv_gamma = 0.05 + 5.0 * torch.sigmoid(self.pv_gamma)
        shunting_factor = 1.0 + safe_pv_gamma * a_pv
        z_normalized = z_integ / shunting_factor

        self.last_pv_shunting = shunting_factor.detach().mean().item()
        
        return self.act(z_normalized)


class CorticalColumn(BasicModel):
    def __init__(
        self,
        num_features: int
    ):
        super().__init__()

        self.num_features = num_features

        self.layer1  = ApicalDendriteIntegration(num_features)
        self.layer23 = L23Integration(num_features, phase_shift=True)
        self.layer5  = PhaseCrossAttention(num_features, use_softmax=False, mode='matrix')
        
    def forward(
        self,
        x_original: torch.Tensor,
        x_pfc: Optional[torch.Tensor],
        x_thalamus_phase: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l1_output: torch.Tensor = self.layer1(
            x_pfc=x_pfc,
            x_thalamus_phase=x_thalamus_phase
        )
        l23_output: torch.Tensor = self.layer23(
            x_original=x_original,
            x_higher=l1_output
        )

        l5_output, attn_weights = self.layer5(
            x_original=x_original,
            x_l23=l23_output,
            x_higher=l1_output
        )

        return l23_output, l5_output # original proj, sematic/higher/pfc proj