from codon.base import *


class PhaseShift(BasicModel):
    def __init__(
        self,
        num_features: int,
        initial: float = 0.15,
        learnable: bool = True,
        dim: int = -1
    ):
        super().__init__()

        self.num_features = num_features
        self.initial = initial
        self.dim = dim

        init_tensor = torch.full((num_features,), float(initial), dtype=torch.float32)
        self.weight = nn.Parameter(init_tensor, requires_grad=learnable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotator = torch.polar(torch.ones_like(self.weight), -self.weight)
        
        ndims = x.dim()
        
        actual_dim = self.dim if self.dim >= 0 else ndims + self.dim
        
        shape = [1] * ndims
        shape[actual_dim] = self.num_features
        
        rotator_broadcast = rotator.view(*shape)
        
        return x * rotator_broadcast