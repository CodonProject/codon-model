from codon.base import *

import math


class ComplexLinear(BasicModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        bound = 1.0 / math.sqrt(2.0 * in_features)

        w_real = torch.empty(out_features, in_features).uniform_(-bound, bound)
        w_imag = torch.empty(out_features, in_features).uniform_(-bound, bound)

        self.weight = nn.Parameter(torch.complex(w_real, w_imag))

        if self.bias:
            b_real = torch.zeros(out_features)
            b_imag = torch.zeros(out_features)
            self.bias_param = nn.Parameter(torch.complex(b_real, b_imag))
        else:
            self.register_parameter('bias_param', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex(): x = x.to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias_param)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}'