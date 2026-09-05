import torch
import torch.nn.functional as F


class _SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
        return grad_input


class _GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865475))
        pdf = 0.3989422804014327 * torch.exp(-0.5 * x * x)
        grad_input = grad_output * (cdf + x * pdf)
        return grad_input


class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.sigmoid(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return grad_output * y * (1.0 - y)


class _Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return grad_output * (1.0 - y * y)


class _ELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        y = F.elu(x, alpha)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = torch.where(y > 0.0, grad_output, grad_output * (y + alpha))
        return grad_input, None


class _Softplus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, threshold):
        ctx.beta = beta
        y = F.softplus(x, beta, threshold)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output * (1.0 - torch.exp(-beta * y))
        return grad_input, None, None


class _LogSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = F.logsigmoid(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return grad_output * (1.0 - torch.exp(y))


class _Mish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        is_large = x > 20
        softplus = torch.where(is_large, x, torch.log1p(torch.exp(x.clamp(max=20))))
        return x * torch.tanh(softplus)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        is_large = x > 20
        sp = torch.where(is_large, x, torch.log1p(torch.exp(x.clamp(max=20))))
        t = torch.tanh(sp)
        sig = torch.sigmoid(x)
        grad_input = grad_output * (t + x * sig * (1.0 - t * t))
        return grad_input


__all__ = [
    '_SiLU', '_GELU', '_Sigmoid', '_Tanh',
    '_ELU', '_Softplus', '_LogSigmoid', '_Mish',
]