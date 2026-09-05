import torch


def is_exporting() -> bool:
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()


__all__ = [
    'is_exporting',
]