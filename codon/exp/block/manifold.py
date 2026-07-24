from codon import *
from codon.block import BasicManifoldLinear


class ExpManifoldLinear(BasicManifoldLinear):
    '''
    An experimental linear layer projecting data onto a Riemannian manifold.
    Unlike RiemannianManifoldLinear, this class assigns an independent 
    kappa and lambda_rate to each manifold anchor (output feature).
    
    Attributes:
        kappa (nn.Parameter): Concentration parameter vector [out_features].
        lambda_rate (nn.Parameter): Gravitational attraction coefficient vector [out_features].
        scale (nn.Parameter): Vector amplifier for the hyperspherical network.
        bias (nn.Parameter): Manifold bias vector.
    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kappa_init: float = 2.0,
        lambda_init: float = 0.1,
        scale_init: float = 15.0,
        k_neighbors: int = 2,
        rule: str = 'near'
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            k_neighbors=k_neighbors
        )

        self.rule = rule.lower()
        if not self.rule in ['far', 'near']:
            raise ValueError(f"Invalid rule: {self.rule}, must be 'far' or 'near'")
        
        self.kappa = nn.Parameter(torch.ones(out_features) * kappa_init)
        self.lambda_rate = nn.Parameter(torch.ones(out_features) * lambda_init)
        
        self.scale = nn.Parameter(torch.ones(out_features) * scale_init)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        Resets the parameters of the layer.
        '''
        nn.init.normal_(self.weight, 0, 0.01)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call using native PyTorch
        to support vectorized kappa and lambda parameters.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output data with shape (batch_size, out_features).
        '''
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(input_tensor, p=2, dim=1)
        c = F.linear(x_norm, w_norm)
        
        c_clamp = torch.clamp(c, -1.0 + 1e-6, 1.0 - 1e-6)
        
        theta = torch.acos(c_clamp)
        
        k = self.kappa.view(1, -1)
        l = torch.clamp(self.lambda_rate.view(1, -1), 1e-6, 1.0 - 1e-4)
        
        exp_val = torch.exp(k * (c_clamp - 1.0))
        if self.rule == 'near':
            attraction = exp_val
        else:
            attraction = 1.0 - exp_val
        
        effective_theta = theta * (1.0 - l * attraction)
        
        out = self.scale.view(1, -1) * torch.cos(effective_theta) + self.bias.view(1, -1)
        
        return out

    def extra_repr(self) -> str:
        main_str = super().extra_repr()
        return f'{main_str}, rule={self.rule}'
