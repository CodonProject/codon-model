from codon import *
from codon.block.activation import get_activation


@dataclass
class ModulationOutput:
    scale: torch.Tensor
    shift: torch.Tensor
    output: torch.Tensor


class Affine(BasicModel):
    def __init__(self, channel):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, channel))
        self.b = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        return x * self.g + self.b


class Modulation(BasicModel):
    def __init__(
        self,
        cond_dim,
        out_dim,
        act: str = 'silu'
    ):
        super().__init__()
        self.net = nn.Sequential(
            get_activation(act),
            nn.Linear(cond_dim, out_dim)
        )

    def forward(self, x, condition):
        scale, shift = self.net(condition).chunk(2, dim=-1)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return ModulationOutput(
            scale=scale,
            shift=shift,
            output=x
        )