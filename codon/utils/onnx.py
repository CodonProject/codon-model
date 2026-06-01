import types
import torch
import torch.nn as nn


def rms_norm_onnx_forward(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    '''
    ONNX-friendly mathematically equivalent forward pass for nn.RMSNorm.

    The variance and reciprocal square root are computed in FP32 regardless of
    the input dtype to preserve numerical stability when the surrounding graph
    runs in FP16. The result is cast back to the input dtype before being
    multiplied by `self.weight`, so the exported ONNX graph contains explicit
    Cast nodes and remains type-consistent end to end.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The normalized output tensor with the same dtype as `x`.
    '''
    eps = self.eps if self.eps is not None else 1e-6
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return x_normed.to(input_dtype) * self.weight


def patch_rms_norm(model: nn.Module) -> None:
    '''
    Recursively patches all nn.RMSNorm modules in a model with an ONNX-compatible forward pass.

    Args:
        model (nn.Module): The PyTorch module to patch.
    '''
    for module in model.modules():
        if isinstance(module, nn.RMSNorm):
            module.forward = types.MethodType(rms_norm_onnx_forward, module)
