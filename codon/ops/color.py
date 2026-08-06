from codon import *
from .tensor_utils import prepare_input_tensor

def rgb_to_gray(img_tensor: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    '''
    Convert RGB image tensor to grayscale.

    Args:
        img_tensor (torch.Tensor): Input image tensor.
        device (str): Computation device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Grayscale image tensor.
    '''
    img_tensor = prepare_input_tensor(img_tensor, device=device)
    if img_tensor.shape[1] == 3:
        return 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    return img_tensor

def rgb_to_lab(img_tensor: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    '''
    Convert RGB image tensor to CIELAB.

    Args:
        img_tensor (torch.Tensor): Input image tensor.
        device (str): Computation device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: CIELAB image tensor of shape (3, H, W).
    '''
    img_tensor = prepare_input_tensor(img_tensor, device=device)
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    mask = img_tensor > 0.04045
    rgb_lin = torch.where(mask, ((img_tensor + 0.055) / 1.055) ** 2.4, img_tensor / 12.92)

    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=img_tensor.device, dtype=img_tensor.dtype)

    rgb_perm = rgb_lin.permute(0, 2, 3, 1)
    xyz = torch.matmul(rgb_perm, M.T)

    xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=img_tensor.device)
    xyz_normalized = xyz / xyz_ref

    delta = 6.0 / 29.0
    mask_xyz = xyz_normalized > (delta ** 3)
    f_xyz = torch.where(mask_xyz, torch.pow(torch.clamp(xyz_normalized, min=1e-8), 1.0 / 3.0),
                        (xyz_normalized / (3 * delta ** 2)) + (4.0 / 29.0))

    L = 116.0 * f_xyz[..., 1:2] - 16.0
    a = 500.0 * (f_xyz[..., 0:1] - f_xyz[..., 1:2])
    b = 200.0 * (f_xyz[..., 1:2] - f_xyz[..., 2:3])

    return torch.cat([L, a, b], dim=-1).squeeze(0).permute(2, 0, 1)
