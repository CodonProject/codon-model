from codon import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from codon.ops import is_exporting


def _reshape_parameter(param: torch.Tensor, ndim: int, channel_first: bool) -> torch.Tensor:
    if channel_first and ndim > 2:
        shape = [1, param.numel()] + [1] * (ndim - 2)
        return param.view(shape)
    return param

HAS_NATIVE_RMSNORM = hasattr(nn, 'RMSNorm')

class _RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps, dim, zero_centered: bool):
        orig_dtype = x.dtype
        x_f32 = x.float()
        
        # 计算均方根 (RMS)
        ms = x_f32.pow(2).mean(dim=dim, keepdim=True)
        rsqrt_ms = torch.rsqrt(ms + eps)
        x_normed = x_f32 * rsqrt_ms
        
        ctx.save_for_backward(x_normed, gamma, rsqrt_ms)
        ctx.dim = dim
        ctx.orig_dtype = orig_dtype
        ctx.zero_centered = zero_centered
        
        # 调整 gamma 形状
        gamma_reshaped = _reshape_parameter(gamma, x.ndim, dim == 1 or dim == -(x.ndim - 1))
        
        # ZCRMS 使用 (1 + gamma)，普通 RMS 直接使用 gamma
        scale = (1.0 + gamma_reshaped) if zero_centered else gamma_reshaped
        return (x_normed * scale).to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_normed, gamma, rsqrt_ms = ctx.saved_tensors
        dim = ctx.dim
        orig_dtype = ctx.orig_dtype
        zero_centered = ctx.zero_centered
        
        grad_output = grad_output.float()
        x_normed = x_normed.float()
        gamma = gamma.float()
        rsqrt_ms = rsqrt_ms.float()
        
        ndim = grad_output.ndim
        norm_dim = dim if dim >= 0 else ndim + dim
        channel_first = (norm_dim == 1 and ndim > 2)
        
        # 1. 计算对 gamma 的梯度
        reduce_dims = list(range(ndim))
        reduce_dims.remove(norm_dim)
        d_gamma = (grad_output * x_normed).sum(dim=reduce_dims)
        
        # 2. 计算对输入 x 的梯度
        gamma_reshaped = _reshape_parameter(gamma, ndim, channel_first)
        scale = (1.0 + gamma_reshaped) if zero_centered else gamma_reshaped
        d_x_normed = grad_output * scale
        
        mean_dot = (d_x_normed * x_normed).mean(dim=dim, keepdim=True)
        dx = rsqrt_ms * (d_x_normed - x_normed * mean_dot)
        
        return dx.to(orig_dtype), d_gamma.to(gamma.dtype), None, None, None


class _L1RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps, dim):
        orig_dtype = x.dtype
        x_f32 = x.float()
        
        # 使用 L1-Norm (绝对值的平均) 代替 L2-Norm
        mean_abs = x_f32.abs().mean(dim=dim, keepdim=True)
        scale = 1.0 / (mean_abs + eps)
        x_normed = x_f32 * scale
        
        ctx.save_for_backward(x_normed, gamma, scale, x)
        ctx.dim = dim
        ctx.orig_dtype = orig_dtype
        
        gamma_reshaped = _reshape_parameter(gamma, x.ndim, dim == 1 or dim == -(x.ndim - 1))
        return (x_normed * gamma_reshaped).to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_normed, gamma, scale, x = ctx.saved_tensors
        dim = ctx.dim
        orig_dtype = ctx.orig_dtype
        
        grad_output = grad_output.float()
        x_normed = x_normed.float()
        gamma = gamma.float()
        scale = scale.float()
        x = x.float()
        
        ndim = grad_output.ndim
        norm_dim = dim if dim >= 0 else ndim + dim
        channel_first = (norm_dim == 1 and ndim > 2)
        
        reduce_dims = list(range(ndim))
        reduce_dims.remove(norm_dim)
        
        d_gamma = (grad_output * x_normed).sum(dim=reduce_dims)
        
        gamma_reshaped = _reshape_parameter(gamma, ndim, channel_first)
        d_x_normed = grad_output * gamma_reshaped
        
        # L1 梯度的导数项 (包含 sign 激活)
        d_mean_abs = (d_x_normed * x_normed).mean(dim=dim, keepdim=True)
        dx = scale * (d_x_normed - torch.sign(x) * d_mean_abs)
        
        return dx.to(orig_dtype), d_gamma.to(gamma.dtype), None, None


class RMSNorm(BasicModel):
    """
    Root Mean Square Normalization (RMSNorm).
    Formula: y = (x / RMS(x)) * gamma
    """
    def __init__(self, d_model: int, eps: float = 1e-6, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        
        if is_exporting():
            orig_dtype = x.dtype
            x_f32 = x.float()
            rms = torch.rsqrt(x_f32.pow(2).mean(dim=dim, keepdim=True) + self.eps)
            x_normed = (x_f32 * rms).to(orig_dtype)
            gamma = _reshape_parameter(self.gamma, x.ndim, self.channel_first)
            return x_normed * gamma.to(orig_dtype)
        
        if HAS_NATIVE_RMSNORM and not self.channel_first:
            return F.rms_norm(x, self.gamma.shape, weight=self.gamma, eps=self.eps)
        else:
            return _RMSNormFunc.apply(x, self.gamma, self.eps, dim, False)


class ZCRMSNorm(BasicModel):
    """
    Zero-Centered Root Mean Square Normalization (ZCRMSNorm).
    Formula: y = (x / RMS(x)) * (1 + gamma)
    """
    def __init__(self, d_model: int, eps: float = 1e-6, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        
        if is_exporting():
            orig_dtype = x.dtype
            x_f32 = x.float()
            rms = torch.rsqrt(x_f32.pow(2).mean(dim=dim, keepdim=True) + self.eps)
            x_normed = (x_f32 * rms).to(orig_dtype)
            gamma = _reshape_parameter(self.gamma, x.ndim, self.channel_first)
            return x_normed * (1.0 + gamma.to(orig_dtype))
            
        return _RMSNormFunc.apply(x, self.gamma, self.eps, dim, True)


class LayerNorm(BasicModel):
    """标准 LayerNorm (支持 Channel-First 和 ONNX 导出自动回退)"""
    def __init__(self, d_model: int, eps: float = 1e-5, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        
        if not self.channel_first and not is_exporting():
            return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
            
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean = x_f32.mean(dim=dim, keepdim=True)
        var = x_f32.var(dim=dim, keepdim=True, unbiased=False)
        x_normed = (x_f32 - mean) * torch.rsqrt(var + self.eps)
        
        weight = _reshape_parameter(self.weight, x.ndim, self.channel_first)
        bias = _reshape_parameter(self.bias, x.ndim, self.channel_first)
        
        return (x_normed.to(orig_dtype) * weight.to(orig_dtype) + bias.to(orig_dtype))


class ZCLayerNorm(BasicModel):
    """零中心化 LayerNorm (初始化权重为0，残差连接更稳定)"""
    def __init__(self, d_model: int, eps: float = 1e-5, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.weight = nn.Parameter(torch.zeros(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean = x_f32.mean(dim=dim, keepdim=True)
        var = x_f32.var(dim=dim, keepdim=True, unbiased=False)
        x_normed = (x_f32 - mean) * torch.rsqrt(var + self.eps)
        
        weight = _reshape_parameter(self.weight, x.ndim, self.channel_first)
        bias = _reshape_parameter(self.bias, x.ndim, self.channel_first)
        
        return (x_normed.to(orig_dtype) * (1.0 + weight.to(orig_dtype)) + bias.to(orig_dtype))


class ScaleNorm(BasicModel):
    """极简缩放归一化 (仅单标量参数进行L2归一化)"""
    def __init__(self, d_model: int, eps: float = 1e-5, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.g = nn.Parameter(torch.tensor(d_model ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        orig_dtype = x.dtype
        x_f32 = x.float()
        
        l2_norm = torch.rsqrt(x_f32.pow(2).sum(dim=dim, keepdim=True) + self.eps)
        x_normed = (x_f32 * l2_norm).to(orig_dtype)
        
        g = _reshape_parameter(self.g, x.ndim, self.channel_first)
        return x_normed * g.to(orig_dtype)


class L1RMSNorm(BasicModel):
    """
    L1-Norm Based RMSNorm (端侧 NPU/FPGA 部署极速版).
    无开根与乘方，推理友好。
    """
    def __init__(self, d_model: int, eps: float = 1e-6, channel_first: bool = False):
        super().__init__()
        self.eps = eps
        self.channel_first = channel_first
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        
        if is_exporting():
            orig_dtype = x.dtype
            x_f32 = x.float()
            mean_abs = x_f32.abs().mean(dim=dim, keepdim=True)
            x_normed = (x_f32 / (mean_abs + self.eps)).to(orig_dtype)
            gamma = _reshape_parameter(self.gamma, x.ndim, self.channel_first)
            return x_normed * gamma.to(orig_dtype)
            
        return _L1RMSNormFunc.apply(x, self.gamma, self.eps, dim)


class FRN(BasicModel):
    """
    Filter Response Normalization (FRN).
    最适合小 Batch Size 的 CV 模型，在通道内的空间维度独立归一化。
    """
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))  # 阈值激活参数 (TLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        if ndim < 3:
            raise ValueError(f"FRN requires input ndim >= 3, but got {ndim}")
            
        orig_dtype = x.dtype
        x_f32 = x.float()
        
        spatial_dims = list(range(2, ndim))
        nu2 = x_f32.pow(2).mean(dim=spatial_dims, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(nu2 + self.eps)
        
        gamma = self.gamma
        beta = self.beta
        tau = self.tau
        if ndim != 4:
            shape = [1, x.size(1)] + [1] * (ndim - 2)
            gamma = gamma.view(shape)
            beta = beta.view(shape)
            tau = tau.view(shape)
            
        y = x_normed * gamma + beta
        return torch.max(y, tau).to(orig_dtype)


class EvoNormS0(BasicModel):
    """
    Evolving Normalization (EvoNorm-S0).
    融合了激活函数与归一化，只支持 Channel First 格式 [B, C, H, W]。
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.v = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"EvoNormS0 only supports 4D input [B, C, H, W], but got {x.ndim}D")
            
        orig_dtype = x.dtype
        x_f32 = x.float()
        
        var = x_f32.pow(2).mean(dim=(2, 3), keepdim=True)
        denominator = torch.sqrt(var + self.eps)
        
        # 分子引入基于可学习参数 v 的 Sigmoid 门控机制
        numerator = x_f32 * torch.sigmoid(x_f32 * self.v)
        
        x_normed = (numerator / denominator).to(orig_dtype)
        return x_normed * self.gamma + self.beta



class FlexibleGroupNorm(BasicModel):
    """
    Flexible Group Normalization (自适应 GroupNorm).
    自动进行轴转换和尺寸重塑，完美支持 Channel-First 和 Channel-Last。
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, channel_first: bool = False):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.channel_first = channel_first
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        ndim = x.ndim
        
        if not self.channel_first:
            shape = x.shape
            b, c = shape[0], shape[-1]
            x_flat = x.view(b, -1, c)
            x_trans = x_flat.transpose(1, 2)  # [B, C, L]
        else:
            x_trans = x
            
        b_trans, c_trans = x_trans.size(0), x_trans.size(1)
        x_norm_input = x_trans.view(b_trans, c_trans, -1)  # 扁平化空间维度以支持 1D/2D/3D
        
        y_flat = F.group_norm(
            x_norm_input, 
            num_groups=self.num_groups, 
            weight=self.weight, 
            bias=self.bias, 
            eps=self.eps
        )
        
        if not self.channel_first:
            y_trans = y_flat.transpose(1, 2)
            return y_trans.view(shape).to(orig_dtype)
        else:
            return y_flat.view_as(x).to(orig_dtype)


class FusedInstanceNorm2d(BasicModel):
    """
    ONNX 融合加速的 InstanceNorm2d。
    利用等价的 GroupNorm 算子，防止在 ONNX 导出时被拆散成细粒度小算子。
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x, 
            num_groups=self.num_features, 
            weight=self.weight, 
            bias=self.bias, 
            eps=self.eps
        )



class ExportableSpectralNorm(BasicModel):
    """
    无 Hook 且对 JIT / ONNX 导出友好的谱归一化 (Spectral Normalization)。
    """
    def __init__(self, module: Union[nn.Module, BasicModel], name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        if not hasattr(self.module, self.name):
            raise AttributeError(f"Module {self.module} has no parameter named {self.name}")

        weight = getattr(self.module, self.name)
        
        self.register_buffer('u', torch.randn(weight.size(0), 1))
        self.u.data = F.normalize(self.u.data, dim=0, eps=self.eps)
        
        delattr(self.module, self.name)
        self.register_parameter(self.name + "_orig", nn.Parameter(weight.data))

    @torch.no_grad()
    def _power_iteration(self, weight_mat: torch.Tensor) -> torch.Tensor:
        u = self.u
        v = None
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        
        if self.training:
            self.u.copy_(u)
        return u, v

    def forward(self, *args, **kwargs) -> torch.Tensor:
        weight_orig = getattr(self, self.name + "_orig")
        weight_mat = weight_orig.view(weight_orig.size(0), -1)
        
        if self.training or is_exporting():
            u, v = self._power_iteration(weight_mat)
        else:
            u = self.u
            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
            
        sigma = torch.matmul(u.t(), torch.matmul(weight_mat, v))
        weight_sn = weight_orig / (sigma + self.eps)
        
        setattr(self.module, self.name, weight_sn)
        return self.module(*args, **kwargs)


class WSConv2d(nn.Conv2d):
    """
    权重标准化卷积 (Weight Standardization Convolution)。
    在卷积计算前对 Filter 进行归一化，通常与 GroupNorm 组合使用。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-5):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride, padding, dilation, groups, bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_mean = w.mean(dim=[1, 2, 3], keepdim=True)
        w_var = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        w_standardized = (w - w_mean) * torch.rsqrt(w_var + self.eps)
        
        return F.conv2d(
            x, w_standardized, self.bias, self.stride, 
            self.padding, self.dilation, self.groups
        )
