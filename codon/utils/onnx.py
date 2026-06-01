import types
import torch
import torch.nn as nn


def rms_norm_onnx_forward(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    '''
    ONNX-friendly mathematically equivalent forward pass for nn.RMSNorm.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The normalized output tensor.
    '''
    eps = self.eps if self.eps is not None else 1e-6
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * self.weight


def patch_rms_norm(model: nn.Module) -> None:
    '''
    Recursively patches all nn.RMSNorm modules in a model with an ONNX-compatible forward pass.

    Args:
        model (nn.Module): The PyTorch module to patch.
    '''
    for module in model.modules():
        if isinstance(module, nn.RMSNorm):
            module.forward = types.MethodType(rms_norm_onnx_forward, module)
