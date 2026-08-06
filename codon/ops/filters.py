from codon import *
from .tensor_utils import prepare_input_tensor

def get_gaussian_kernel_2d(kernel_size: int = 5, sigma: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    '''
    Generate a 2D Gaussian kernel tensor.

    Args:
        kernel_size (int): Size of the Gaussian kernel. Defaults to 5.
        sigma (float): Standard deviation. Defaults to 1.0.
        device (str): Device to allocate kernel on. Defaults to 'cpu'.

    Returns:
        torch.Tensor: 2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size).
    '''
    pad = kernel_size // 2
    x = torch.arange(kernel_size, device=device) - pad
    grid = x.repeat(kernel_size, 1)
    kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    return kernel

def gaussian_blur_2d(img_tensor: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    '''
    Apply 2D Gaussian blur to the input image tensor.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
        sigma (float): Standard deviation. Defaults to 1.0.
        kernel_size (int): Size of the filter kernel. Defaults to 5.

    Returns:
        torch.Tensor: Blurred image tensor.
    '''
    pad = kernel_size // 2
    kernel = get_gaussian_kernel_2d(kernel_size=kernel_size, sigma=sigma, device=img_tensor.device)
    C = img_tensor.shape[1]
    if C > 1:
        kernel = kernel.repeat(C, 1, 1, 1)
        return F.conv2d(img_tensor, kernel, padding=pad, groups=C)
    return F.conv2d(img_tensor, kernel, padding=pad)

def compute_image_gradients(
    img_tensor: torch.Tensor,
    blur_sigma: float = 0.0,
    kernel_size: int = 5,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Compute image gradient magnitude, x/y gradients, and angles.

    Supports automatic grayscale conversion and optional Gaussian blur.

    Args:
        img_tensor (torch.Tensor): Input image tensor.
        blur_sigma (float): Gaussian blur standard deviation. Disable blur if <= 0. Defaults to 0.0.
        kernel_size (int): Gaussian blur kernel size. Defaults to 5.
        device (str): Computation device. Defaults to 'cpu'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            magnitude, gx, gy, angle
    '''
    img_tensor = prepare_input_tensor(img_tensor, device=device)

    # 1. RGB to Gray
    if img_tensor.shape[1] == 3:
        img_tensor = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

    # 2. Optional Gaussian blur
    if blur_sigma > 0:
        img_tensor = gaussian_blur_2d(img_tensor, sigma=blur_sigma, kernel_size=kernel_size)

    # 3. Sobel Convolution
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    gx = F.conv2d(img_tensor, sobel_x, padding=1).squeeze(0).squeeze(0)
    gy = F.conv2d(img_tensor, sobel_y, padding=1).squeeze(0).squeeze(0)

    magnitude = torch.sqrt(gx**2 + gy**2)
    angle = torch.atan2(gy, gx)

    return magnitude, gx, gy, angle
