from codon.base import *
from codon.ops.complex import *


class ComplexReLU(BasicModel):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return complex_relu(x, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        return f'inplace={self.inplace}'


class ComplexSiLU(BasicModel):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return complex_silu(x)


class ComplexSigmoid(BasicModel):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return complex_sigmoid(x)


class ModReLU(BasicModel):
    def __init__(self, num_features: int, initial_bias: float = -0.01):
        super().__init__()

        self.num_features = num_features
        
        self.bias = nn.Parameter(torch.full((num_features,), initial_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        
        if x.dim() > 2:
            if x.shape[1] == self.num_features:
                ndim = x.dim()
                shape = [1] * ndim
                shape[1] = self.num_features
                bias = bias.view(shape)
            elif x.shape[-1] == self.num_features: pass
        
        return mod_relu(x, bias)
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}'


class ModSiLU(BasicModel):
    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias

        if x.dim() > 2:
            if x.shape[1] == self.num_features:
                ndim = x.dim()
                shape = [1] * ndim
                shape[1] = self.num_features
                bias = bias.view(shape)
        
        return mod_silu(x, bias)
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}'


class ModSigmoid(BasicModel):
    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias

        if x.dim() > 2:
            if x.shape[1] == self.num_features:
                ndim = x.dim()
                shape = [1] * ndim
                shape[1] = self.num_features
                bias = bias.view(shape)
        
        return mod_sigmoid(x, bias)
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}'