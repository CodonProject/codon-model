from codon.base import *
from typing import Literal

import math


class ComplexLinear(BasicModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bound: Literal['natural', 'positive', 'negative'] = 'natural'
    ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.bias         = bias
        self.bound        = bound

        init_bound = 1.0 / math.sqrt(2.0 * in_features)

        w_real = torch.empty(out_features, in_features).uniform_(-init_bound, init_bound)
        w_imag = torch.empty(out_features, in_features).uniform_(-init_bound, init_bound)
        
        self.weight = nn.Parameter(torch.complex(w_real, w_imag))

        if self.bias:
            b_real = torch.zeros(out_features)
            b_imag = torch.zeros(out_features)
            self.bias_param = nn.Parameter(torch.complex(b_real, b_imag))
        else:
            self.register_parameter('bias_param', None)

    def _bounded_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bound == 'natural':
            return self.weight, self.bias_param

        if self.bound == 'positive':
            w = torch.complex(
                torch.clamp(self.weight.real, min=0.0),
                torch.clamp(self.weight.imag, min=0.0)
            )
            b = None
            if self.bias_param is not None:
                b = torch.complex(
                    torch.clamp(self.bias_param.real, min=0.0),
                    torch.clamp(self.bias_param.imag, min=0.0)
                )
            return w, b

        if self.bound == 'negative':
            w = torch.complex(
                torch.clamp(self.weight.real, max=0.0),
                torch.clamp(self.weight.imag, max=0.0)
            )
            b = None
            if self.bias_param is not None:
                b = torch.complex(
                    torch.clamp(self.bias_param.real, max=0.0),
                    torch.clamp(self.bias_param.imag, max=0.0)
                )
            return w, b

        raise ValueError(f"unknown bound: {self.bound}")

    @torch.no_grad()
    def project(self):
        if self.bound == 'positive':
            self.weight.real.clamp_(min=0.0)
            self.weight.imag.clamp_(min=0.0)
            if self.bias_param is not None:
                self.bias_param.real.clamp_(min=0.0)
                self.bias_param.imag.clamp_(min=0.0)
        elif self.bound == 'negative':
            self.weight.real.clamp_(max=0.0)
            self.weight.imag.clamp_(max=0.0)
            if self.bias_param is not None:
                self.bias_param.real.clamp_(max=0.0)
                self.bias_param.imag.clamp_(max=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(self.weight.dtype)
        w, b = self._bounded_weight()
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias}, bound={self.bound}')