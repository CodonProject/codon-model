from codon.base import *
from codon.block.complex import ComplexLinear, ModReLU, PhaseShift
from typing import Optional, Literal
import math


class ApicalGate(BasicModel):
    def __init__(self):
        super().__init__()
        # SST
        self.sst_base = nn.Parameter(torch.tensor(0.8))
        self.vip_to_sst = nn.Parameter(torch.tensor(1.2))
        # NDNF
        self.ndnf_base = nn.Parameter(torch.tensor(0.5))
        self.ach_to_ndnf = nn.Parameter(torch.tensor(1.5))

    def forward(self, x_pfc: torch.Tensor, ach: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_vip = 5.0 * torch.sigmoid(self.vip_to_sst)
        w_ach = 5.0 * torch.sigmoid(self.ach_to_ndnf)
        
        sst_base = 3.0 * torch.sigmoid(self.sst_base)
        ndnf_base = 3.0 * torch.sigmoid(self.ndnf_base)

        # VIP -> SST
        a_vip = torch.mean(torch.abs(x_pfc), dim=-1, keepdim=True)
        g_sst = torch.sigmoid(sst_base - w_vip * a_vip)
        
        # ACh -> NDNF
        if ach is None: ach = torch.zeros_like(g_sst)
        elif ach.ndim == 1: ach = ach.unsqueeze(-1)
        g_ndnf = torch.sigmoid(ndnf_base - w_ach * ach)
        
        total_gate = (1.0 - g_sst) * (1.0 - g_ndnf)
        return total_gate


class ApicalPreProcessor(BasicModel):
    def __init__(
        self,
        num_features: int,
        proj: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.proj = proj

        self.pfc = ComplexLinear(in_features=num_features, out_features=num_features) if proj else nn.Identity()
        self.gate = ApicalGate()

        self.nmda_threshold = nn.Parameter(torch.tensor(0.5))
        self.nmda_gain = nn.Parameter(torch.tensor(2.0))
        
    def forward(
        self,
        x_pfc: torch.Tensor,
        x_ach: Optional[torch.Tensor] = None,
        x_thalamus_phase: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        z_pfc: torch.Tensor = self.pfc(x_pfc)
        
        amplitude = torch.abs(z_pfc)
        phase = torch.angle(z_pfc)
        
        gate: torch.Tensor = self.gate(x_pfc, x_ach)
        gated_amplitude = amplitude * gate

        safe_threshold = 2.0 * torch.sigmoid(self.nmda_threshold)
        safe_gain      = 5.0 * torch.sigmoid(self.nmda_gain)
        
        nmda_boost = torch.sigmoid(gated_amplitude - safe_threshold) * safe_gain
        boosted_amplitude = gated_amplitude + nmda_boost

        z_apical = torch.polar(boosted_amplitude, phase)
        
        if x_thalamus_phase is not None:
            theta_thalamus = torch.angle(x_thalamus_phase) if x_thalamus_phase.is_complex() else x_thalamus_phase
            phase_diff = phase - theta_thalamus
            
            thalamic_gate = 0.5 * (torch.cos(phase_diff) + 1.0)
            z_apical = z_apical * thalamic_gate

        return z_apical

class SpikeFrequencyAdaptation(BasicModel):
    def __init__(self, num_features: int, tau_init: float = 0.9):
        super().__init__()

        self.num_features = num_features
        
        self.raw_tau = nn.Parameter(torch.tensor(math.log(tau_init / (1.0 - tau_init))))
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        
        self.register_buffer('trace', torch.zeros(1, num_features))
        
    def reset_state(self):
        self.trace.fill_(0.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        amp = z.abs()
        
        tau = torch.sigmoid(self.raw_tau)
        gain = F.softplus(self.raw_gain)
        
        if self.trace.shape[0] != z.shape[0]:
            self.trace = torch.zeros(z.shape[0], self.num_features, dtype=amp.dtype, device=amp.device)
        
        self.trace = tau * self.trace + (1.0 - tau) * amp.detach()
        
        factor = 1.0 / (1.0 + gain * self.trace)
            
        return z * factor


class BasalIntegration(BasicModel):
    def __init__(
        self,
        num_features: int,
        phase_shift: bool = True,
        proj: bool = True,
        pv_ratio: float = 0.25
    ):
        super().__init__()

        self.num_features = num_features
        self.use_phase_shift = phase_shift
        self.proj = proj

        self.shift = PhaseShift(num_features) if phase_shift else nn.Identity()

        self.ff_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False, bound='positive') if proj else nn.Identity()
        self.fb_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False, bound='positive') if proj else nn.Identity()

        self.apical = nn.Parameter(torch.tensor(0.2))

        self.num_pv = max(1, int(num_features * pv_ratio))
        self.pyr_to_pv = ComplexLinear(in_features=num_features, out_features=self.num_pv, bias=False, bound='positive')
        self.pv_act = ModReLU(self.num_pv)
        self.pv_to_pyr = ComplexLinear(in_features=self.num_pv, out_features=num_features, bias=False, bound='positive')
        self.pv_gamma = nn.Parameter(torch.tensor(0.1))
        self.act = ModReLU(num_features)

        self.sfa = SpikeFrequencyAdaptation(num_features)
    
    def cortex_scaling(self):
        with torch.no_grad():
            for p in self.act.parameters(): p.clamp_(max=0.0)
            for p in self.pv_act.parameters(): p.clamp_(max=0.0)
    
    def forward(
        self,
        x_original: torch.Tensor,
        x_apical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        x_original: torch.Tensor = self.shift(x_original)
        
        z_ff: torch.Tensor = self.ff_proj(x_original)
        
        if x_apical is not None:
            z_fb: torch.Tensor = self.fb_proj(x_apical)
            z_pyr = z_ff + z_fb * torch.sigmoid(self.apical)
        else:
            z_pyr = z_ff
        
        z_pv: torch.Tensor = self.pv_act(self.pyr_to_pv(x_original))
        
        pv_conductance = torch.abs(self.pv_to_pyr(z_pv))
        
        safe_pv_gamma = 0.05 + 5.0 * torch.sigmoid(self.pv_gamma)
        shunting_factor = 1.0 + safe_pv_gamma * pv_conductance
        
        z_normalized = z_pyr / shunting_factor
        z_activated = self.act(z_normalized)

        return self.sfa(z_activated)


class BACIntegration(BasicModel):
    def __init__(
        self,
        num_features: int,
        work_type: Literal['ctc', 'pre'] = 'pre',
        proj: bool = True
    ):
        super().__init__()

        assert work_type in ['ctc', 'pre']

        self.num_features = num_features
        self.work_type = work_type
        self.proj = proj

        self.q_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if proj else nn.Identity()
        self.k_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if proj else nn.Identity()
        self.v_proj = ComplexLinear(in_features=num_features, out_features=num_features, bias=False) if proj else nn.Identity()
        
        # BAC
        self.apical_threshold = nn.Parameter(torch.tensor(-0.5))
        self.basal_threshold = nn.Parameter(torch.tensor(-0.5))
        self.bac_slope = nn.Parameter(torch.tensor(1.0))

        self.beta_single = nn.Parameter(torch.tensor(0.3))
        self.alpha_res = nn.Parameter(torch.tensor(0.4))
    
    def forward(
        self,
        x_apical: torch.Tensor,
        x_basal: torch.Tensor,
        x_original: torch.Tensor,
        x_thalamus_phase: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        x_apical: torch.Tensor     = self.q_proj(x_apical)
        x_basal_proj: torch.Tensor = self.k_proj(x_basal)
        x_original: torch.Tensor   = self.v_proj(x_original)

        numerator = (x_apical * x_basal_proj.conj()).real
        denominator = torch.abs(x_apical) * torch.abs(x_basal_proj) + 1e-8
        cos_diff = numerator / denominator

        if self.work_type == 'ctc':
            coherence_gate = torch.relu(cos_diff)
        elif self.work_type == 'pre':
            coherence_gate = cos_diff
        else:
            raise ValueError(f'unknown work_type: {self.work_type}')

        safe_apical_thresh = 2.0 * torch.sigmoid(self.apical_threshold)
        safe_basal_thresh  = 2.0 * torch.sigmoid(self.basal_threshold)

        amp_apical = torch.abs(x_apical)
        amp_basal  = torch.abs(x_basal_proj)

        slope = 1.0 + 4.0 * torch.sigmoid(self.bac_slope)
        apical_active = torch.sigmoid(slope * (amp_apical - safe_apical_thresh))
        basal_active  = torch.sigmoid(slope * (amp_basal - safe_basal_thresh))
        amp_gate = apical_active * basal_active
        
        coincidence = coherence_gate * amp_gate

        if x_thalamus_phase is not None:
            theta_thal = torch.angle(x_thalamus_phase) if x_thalamus_phase.is_complex() else x_thalamus_phase
            thal_gate = 0.5 * (torch.cos(torch.angle(x_original) - theta_thal) + 1.0)
            coincidence = coincidence * thal_gate
        
        coincidence_complex = torch.complex(coincidence, torch.zeros_like(coincidence))
        
        z_burst = coincidence_complex * x_original
        
        # Single-spike
        safe_beta = torch.sigmoid(self.beta_single)
        z_single = (1.0 - coincidence_complex) * x_original * safe_beta
        
        # Dendritic bypass
        safe_alpha = torch.sigmoid(self.alpha_res)
        z_l5_output = (z_burst + z_single) + safe_alpha * x_basal
        
        return z_l5_output, coincidence


class CorticalColumn(BasicModel):
    def __init__(
        self,
        num_features: int,
        apical_proj: bool = True,
        basal_proj: bool = True,
        bac_proj: bool = True,
        phase_shift: bool = True,
        work_type: Literal['ctc', 'pre'] = 'pre',
        track: bool = True
    ):
        super().__init__()

        self.num_features = num_features
        self.track = track

        self.apical = ApicalPreProcessor(num_features, proj=apical_proj)
        self.l23    = BasalIntegration(num_features, phase_shift=phase_shift, proj=basal_proj)
        self.l5     = BACIntegration(num_features, proj=bac_proj, work_type=work_type)

        self.last_apical_amp  = None
        self.last_l23_amp     = None
        self.last_l5_amp      = None
        self.last_coincidence = None

    def _zero_pfc(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ref.shape[0], self.num_features, dtype=ref.dtype, device=ref.device)

    def forward(
        self,
        x_l4: torch.Tensor,
        x_pfc: Optional[torch.Tensor] = None,
        x_thalamus_phase: Optional[torch.Tensor] = None,
        x_ach: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_pfc is None: x_pfc = self._zero_pfc(x_l4)

        z_apical = self.apical(
            x_pfc=x_pfc,
            x_ach=x_ach,
            x_thalamus_phase=x_thalamus_phase
        )

        z_l23 = self.l23(
            x_original=x_l4,
            x_apical=z_apical
        )

        z_l5, coincidence = self.l5(
            x_apical=z_apical,
            x_basal=z_l23,
            x_original=x_l4,
            x_thalamus_phase=x_thalamus_phase
        )

        if self.track:
            with torch.no_grad():
                self.last_apical_amp  = z_apical.abs().mean().item()
                self.last_l23_amp     = z_l23.abs().mean().item()
                self.last_l5_amp      = z_l5.abs().mean().item()
                self.last_coincidence = coincidence.mean().item()

        return z_l23, z_l5