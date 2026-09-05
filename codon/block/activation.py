from codon import *
import inspect

from codon.ops import is_exporting
from codon.ops.activation import (
    _ELU,
    _GELU,
    _LogSigmoid,
    _Mish,
    _Sigmoid,
    _SiLU,
    _Softplus,
    _Tanh,
)


def _safe_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not is_exporting() and a.dtype == torch.float16:
        a_min, a_max = torch.aminmax(a.detach())
        b_min, b_max = torch.aminmax(b.detach())
        max_abs_a = torch.max(-a_min, a_max).float()
        max_abs_b = torch.max(-b_min, b_max).float()
        max_abs_value = max_abs_a * max_abs_b
        if max_abs_value > 1000:
            ratio = (1000 / max_abs_value).half()
            b = b * ratio
            return (a * b).clamp(-1000 * ratio, 1000 * ratio) / ratio
    
    return a * b


# 2. 优化后的标准激活函数模块

class SiLU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return x * torch.sigmoid(x)
        return _SiLU.apply(x)


class GELU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.gelu(x)
        return _GELU.apply(x)


class Sigmoid(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return torch.sigmoid(x)
        return _Sigmoid.apply(x)


class Tanh(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return torch.tanh(x)
        return _Tanh.apply(x)


class ELU(BasicModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.elu(x, self.alpha)
        return _ELU.apply(x, self.alpha)


class CELU(BasicModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.celu(x, self.alpha)
        # CELU(x) = ELU(x / alpha) * alpha
        return _ELU.apply(x / self.alpha, 1.0) * self.alpha


class SELU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.selu(x)
        # SELU 是 Scale * ELU
        scale = 1.0507009873554805
        alpha = 1.6732632423543085
        return _ELU.apply(x, alpha) * scale


class Softplus(BasicModel):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.softplus(x, self.beta, self.threshold)
        return _Softplus.apply(x, self.beta, self.threshold)


class LogSigmoid(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.logsigmoid(x)
        return _LogSigmoid.apply(x)


class Mish(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return x * torch.tanh(F.softplus(x))
        return _Mish.apply(x)


# 3. 精度安全与推理优化函数 (Softmax & ONNX Fusion)

class Softmax(BasicModel):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.softmax(x, dim=self.dim)
        if x.dtype in (torch.float16, torch.bfloat16):
            return F.softmax(x.float(), dim=self.dim).to(dtype=x.dtype)
        return F.softmax(x, dim=self.dim)


class Softmin(BasicModel):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.softmin(x, dim=self.dim)
        if x.dtype in (torch.float16, torch.bfloat16):
            return F.softmin(x.float(), dim=self.dim).to(dtype=x.dtype)
        return F.softmin(x, dim=self.dim)


class LogSoftmax(BasicModel):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.log_softmax(x, dim=self.dim)
        if x.dtype in (torch.float16, torch.bfloat16):
            return F.log_softmax(x.float(), dim=self.dim).to(dtype=x.dtype)
        return F.log_softmax(x, dim=self.dim)


class Softmax2d(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return F.softmax(x, dim=1)
        if x.dtype in (torch.float16, torch.bfloat16):
            return F.softmax(x.float(), dim=1).to(dtype=x.dtype)
        return F.softmax(x, dim=1)


class Hardshrink(BasicModel):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            mask = (x > self.lambd) | (x < -self.lambd)
            return x * mask.to(x.dtype)
        return F.hardshrink(x, self.lambd)


class Softshrink(BasicModel):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return torch.sign(x) * torch.clamp(x.abs() - self.lambd, min=0)
        return F.softshrink(x, self.lambd)


class Hardtanh(BasicModel):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)


class LeakyReLU(BasicModel):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)


class PReLU(BasicModel):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            weight = self.weight
            if x.ndim == 4 and weight.numel() > 1:
                weight = weight.view(1, -1, 1, 1)
            elif x.ndim == 3 and weight.numel() > 1:
                weight = weight.view(1, 1, -1)
            return torch.where(x > 0, x, x * weight)
        return F.prelu(x, self.weight)


class RReLU(BasicModel):
    def __init__(self, lower: float = 1.0 / 8.0, upper: float = 1.0 / 3.0):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting() or not self.training:
            return F.leaky_relu(x, (self.lower + self.upper) / 2.0)
        return F.rrelu(x, self.lower, self.upper, training=True)


class ReLU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class ReLU6(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x)


class Softsign(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return x / (1.0 + x.abs())
        return F.softsign(x)


class Tanhshrink(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.tanh(x)


class Threshold(BasicModel):
    def __init__(self, threshold: float, value: float):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_exporting():
            return torch.where(x > self.threshold, x, torch.tensor(self.value, dtype=x.dtype, device=x.device))
        return F.threshold(x, self.threshold, self.value)


# 4. GLU 家族 & 之前编写的激活函数

class SwiGLU(BasicModel):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return _safe_product(out, F.silu(gate))


class GeGLU(BasicModel):
    def __init__(self, dim=-1, approximate: str = 'none'):
        super().__init__()
        self.dim = dim
        self.approximate = approximate

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.gelu(gate, approximate=self.approximate)
        return _safe_product(out, gate)


class ReGLU(BasicModel):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return _safe_product(out, F.relu(gate))


class GLU(BasicModel):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return _safe_product(out, torch.sigmoid(gate))


class QuickGELU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class HardSwish(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=False) / 6.0


class HardSigmoid(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=False) / 6.0


class SquaredReLU(BasicModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(F.relu(x), 2)


class StarReLU(BasicModel):
    def __init__(self, dim: int = None, use_scale: bool = True, use_bias: bool = True):
        super().__init__()
        if dim is not None:
            self.scale = nn.Parameter(torch.ones(dim)) if use_scale else 1.0
            self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else 0.0
        else:
            self.scale = nn.Parameter(torch.ones(1)) if use_scale else 1.0
            self.bias = nn.Parameter(torch.zeros(1)) if use_bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        bias = self.bias
        if isinstance(scale, nn.Parameter) and x.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
        if isinstance(bias, nn.Parameter) and x.ndim == 4:
            bias = bias.view(1, -1, 1, 1)
            
        return torch.pow(F.relu(x), 2) * scale + bias


_ACT_ALIAS: Dict[str, str] = {
    'swish': 'silu',
    'gated-linear-unit': 'glu',
    'leaky': 'leakyrelu',
}

def get_activation(name: Union[str, BasicModel], **kwargs) -> BasicModel:
    if isinstance(name, BasicModel):
        return name
        
    if not isinstance(name, str):
        raise TypeError(f'Activation name must be a string or BasicModel, but got {type(name)}')

    raw_name = name
    name = name.lower().replace('_', '').replace('-', '')
    
    if name in _ACT_ALIAS:
        name = _ACT_ALIAS[name]
        
    act_map: Dict[str, Type[BasicModel]] = {
        'relu': ReLU,
        'relu6': ReLU6,
        'silu': SiLU,
        'gelu': GELU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'mish': Mish,
        'elu': ELU,
        'celu': CELU,
        'selu': SELU,
        'softplus': Softplus,
        'logsigmoid': LogSigmoid,
        'softmax': Softmax,
        'softmin': Softmin,
        'logsoftmax': LogSoftmax,
        'softmax2d': Softmax2d,
        'hardshrink': Hardshrink,
        'softshrink': Softshrink,
        'hardtanh': Hardtanh,
        'leakyrelu': LeakyReLU,
        'prelu': PReLU,
        'rrelu': RReLU,
        'softsign': Softsign,
        'tanhshrink': Tanhshrink,
        'threshold': Threshold,
        
        'swiglu': SwiGLU,
        'geglu': GeGLU,
        'reglu': ReGLU,
        'glu': GLU,
        
        'quickgelu': QuickGELU,
        'hardswish': HardSwish,
        'hardsigmoid': HardSigmoid,
        'squaredrelu': SquaredReLU,
        'starrelu': StarReLU,
    }
    
    if name not in act_map:
        raise ValueError(
            f"Unsupported activation: '{raw_name}'."
            f'Supported activations are: {list(act_map.keys()) + list(_ACT_ALIAS.keys())}'
        )
        
    act_class = act_map[name]
    
    sig = inspect.signature(act_class.__init__)
    valid_params = sig.parameters.keys()
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return act_class(**filtered_kwargs)