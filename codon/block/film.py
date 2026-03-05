from codon.base import *

from dataclasses import dataclass
from typing      import Optional

@dataclass
class FiLMOutput:
    '''
    FiLM module output container.
    
    Attributes:
        output (torch.Tensor): Features modulated by gamma and beta.
        gate (Optional[torch.Tensor]): Gating values for residual connections.
    '''
    output: torch.Tensor
    gate: Optional[torch.Tensor] = None

    @property
    def gated_output(self):
        if self.gate is None: return self.output
        return self.output * self.gate


class FiLM(BasicModel):
    '''
    Feature-wise Linear Modulation (FiLM) module.

    Applies affine transformation to input features: FiLM(x) = (1 + gamma(z)) * x + beta(z)
    where gamma and beta are generated from the conditional input z.
    Initially, gamma is 0 and beta is 0, resulting in an identity mapping.

    Attributes:
        proj (nn.Linear, optional): Linear layer to project conditional features to gamma, beta, and gate.
        gate_proj (nn.Module): Linear layer or Identity used for context gating.
    '''
    def __init__(
        self,
        in_features: int,
        cond_features: int,
        use_beta: bool = True,
        use_gamma: bool = True,
        use_gate: bool = True,
        use_context_gate: bool = False,
        channel_first: bool = False
    ):
        '''
        Initialize the FiLM module.

        Args:
            in_features (int): Dimension of input features.
            cond_features (int): Dimension of conditional features.
            use_beta (bool, optional): Whether to use the translation term (beta). Defaults to True.
            use_gamma (bool, optional): Whether to use the scaling term (gamma). Defaults to True.
            use_gate (bool, optional): Whether to use the gating term (gate). Defaults to True.
            use_context_gate (bool, optional): Whether to use context gating.
                If True, uses the concatenation of input features and conditional features to generate gating values, overriding use_gate setting. Defaults to False.
            channel_first (bool, optional): Whether the feature dimension is the 1st dimension (e.g., CNN [B, C, H, W]).
                If False, assumes features are in the last dimension (e.g., Transformer [B, L, C]). Defaults to False.
        '''
        super().__init__()
        
        if use_context_gate: use_gate = False

        self.in_features = in_features
        self.cond_features = cond_features
        self.use_beta = use_beta
        self.use_gamma = use_gamma
        self.use_gate = use_gate
        self.use_context_gate = use_context_gate
        self.channel_first = channel_first

        self.out_dim = 0
        if use_gamma: self.out_dim += in_features
        if use_beta:  self.out_dim += in_features
        if use_gate:  self.out_dim += in_features

        self.gate_proj = nn.Linear(in_features + cond_features, in_features) if use_context_gate else nn.Identity()

        if self.out_dim > 0:
            self.proj = nn.Linear(cond_features, self.out_dim)
        else: self.proj = None

        self._init_weights(self)
    
    def _init_weights(self, model: nn.Module):
        '''
        Initializes weights.

        Initializes weights and biases of projection layers to 0 to ensure initial identity mapping.
        If context gating is used, its projection layer uses Xavier Uniform initialization.

        Args:
            model (nn.Module): The model to initialize.
        '''
        if model is self and self.proj is not None:
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)
            if isinstance(self.gate_proj, nn.Identity): return
            nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
            nn.init.zeros_(self.gate_proj.bias)

    def _reshape(self, param: torch.Tensor, ref_ndim: int) -> torch.Tensor:
        '''
        Reshapes parameter to match input feature dimensions for broadcasting.

        Args:
            param (torch.Tensor): Parameter tensor to reshape.
            ref_ndim (int): Number of dimensions of the reference tensor (usually input features x).

        Returns:
            torch.Tensor: Reshaped parameter tensor.
        '''
        if self.channel_first:
            param = param.movedim(-1, 1)
            for _ in range(ref_ndim - param.ndim):
                param = param.unsqueeze(-1)
        else:
            for _ in range(ref_ndim - param.ndim):
                param = param.unsqueeze(-2)
        return param

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> FiLMOutput:
        '''
        Forward pass.

        Args:
            x (torch.Tensor): Input features. Shape: [B, C, ...] (if channel_first=True)
                or [B, ..., C] (if channel_first=False).
            cond (torch.Tensor): Conditional input. Shape: [B, ..., cond_features].

        Returns:
            FiLMOutput: Modulated features.
        '''
        if self.proj is None: return FiLMOutput(output=x)
        
        params = self.proj(cond)
        
        count = sum([self.use_gamma, self.use_beta, self.use_gate])
        if count > 1:
            params_list = params.chunk(count, dim=-1)
        else:
            params_list = [params]
        
        idx = 0
        gamma, beta, gate = None, None, None
        if self.use_gamma:
            gamma = params_list[idx]
            idx += 1
        if self.use_beta:
            beta = params_list[idx]
            idx += 1
        if self.use_gate:
            gate = params_list[idx]
            idx += 1
        
        out = x
        if gamma is not None:
            out = out * (1 + self._reshape(gamma, x.ndim))
        if beta is not None:
            out = out + self._reshape(beta, x.ndim)
        
        final_gate = None
        if self.use_context_gate:
            if cond.ndim < x.ndim:
                shape = list(x.shape)
                feat_dim = 1 if self.channel_first else -1
                shape[feat_dim] = -1
                cond_expanded = self._reshape(cond, x.ndim).expand(shape)
            else:
                cond_expanded = cond
            
            feat_dim = 1 if self.channel_first else -1
            context_input = torch.cat([x, cond_expanded], dim=feat_dim)
            
            if self.channel_first:
                context_input = context_input.movedim(1, -1)
                final_gate = self.gate_proj(context_input).movedim(-1, 1)
            else:
                final_gate = self.gate_proj(context_input)
        
        elif gate is not None:
            final_gate = self._reshape(gate, x.ndim)
        
        return FiLMOutput(output=out, gate=final_gate)
