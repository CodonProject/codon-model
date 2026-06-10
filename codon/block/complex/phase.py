from codon.base import *


class PhaseShift(BasicModel):
    def __init__(
        self,
        num_features: int
    ):
        super().__init__()

        self.num_features = num_features
        self.weight = nn.Parameter(torch.zeros(num_features) + 0.15)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotator = torch.polar(torch.ones_like(self.weight), -self.weight)
        return x * rotator

