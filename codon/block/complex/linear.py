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

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.bound = bound

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
    
    @torch.no_grad()
    def enforce_positive(self):
        clamped_w_real = torch.clamp(self.weight.real, min=0.0)
        clamped_w_imag = torch.clamp(self.weight.imag, min=0.0)
        self.weight.copy_(torch.complex(clamped_w_real, clamped_w_imag))
        
        if self.bias_param is not None:
            clamped_b_real = torch.clamp(self.bias_param.real, min=0.0)
            clamped_b_imag = torch.clamp(self.bias_param.imag, min=0.0)
            self.bias_param.copy_(torch.complex(clamped_b_real, clamped_b_imag))
    
    @torch.no_grad()
    def enforce_negative(self):
        clamped_w_real = torch.clamp(self.weight.real, max=0.0)
        clamped_w_imag = torch.clamp(self.weight.imag, max=0.0)
        self.weight.copy_(torch.complex(clamped_w_real, clamped_w_imag))
        
        if self.bias_param is not None:
            clamped_b_real = torch.clamp(self.bias_param.real, max=0.0)
            clamped_b_imag = torch.clamp(self.bias_param.imag, max=0.0)
            self.bias_param.copy_(torch.complex(clamped_b_real, clamped_b_imag))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bound == 'positive': self.enforce_positive()
        if self.bound == 'negative': self.enforce_negative()

        if not x.is_complex(): x = x.to(self.weight.dtype)
        return F.linear(x, self.weight, self.bias_param)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}, bound={self.bound}'