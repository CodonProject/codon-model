from codon import *

def prepare_input_tensor(img: Union[np.ndarray, torch.Tensor], device: str = 'cpu') -> torch.Tensor:
    '''
    Format input array or tensor into (1, C, H, W) float32 Tensor and move to device.

    Args:
        img (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        device (str): Computation device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Formatted image tensor of shape (1, C, H, W).
    '''
    if isinstance(img, np.ndarray):
        tensor = torch.from_numpy(img)
    else:
        tensor = img

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        if tensor.shape[2] in [1, 3]:  # HWC -> CHW
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

    return tensor.to(device).float()
