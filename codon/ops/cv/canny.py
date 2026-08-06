from codon import *
from codon.ops import compute_image_gradients


def preprocess_canny_pytorch(
    img_tensor: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.4,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Preprocess image using PyTorch to calculate gradient magnitude and angles.

    Apply Gaussian blur and Sobel filters to calculate image gradients.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W) or (1, C, H, W).
        kernel_size (int): Gaussian blur kernel size.
        sigma (float): Gaussian blur standard deviation.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradient magnitude and angles as NumPy arrays.
    '''
    magnitude, _, _, angle = compute_image_gradients(
        img_tensor, blur_sigma=sigma, kernel_size=kernel_size, device=device
    )
    return magnitude.cpu().numpy(), angle.cpu().numpy()

@numba.jit(nopython=True, fastmath=True)
def _non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    '''
    Perform non-maximum suppression to thin out edges.

    Args:
        mag (np.ndarray): Gradient magnitude array.
        angle (np.ndarray): Gradient angle array.

    Returns:
        np.ndarray: Non-maximum suppressed magnitude array.
    '''
    H, W = mag.shape
    nms = np.zeros((H, W), dtype=np.float32)
    
    angle_deg = np.rad2deg(angle) % 180.0

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            ang = angle_deg[y, x]
            m = mag[y, x]
            
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = mag[y, x + 1]
                r = mag[y, x - 1]
            elif 22.5 <= ang < 67.5:
                q = mag[y - 1, x + 1]
                r = mag[y + 1, x - 1]
            elif 67.5 <= ang < 112.5:
                q = mag[y - 1, x]
                r = mag[y + 1, x]
            elif 112.5 <= ang < 157.5:
                q = mag[y - 1, x - 1]
                r = mag[y + 1, x + 1]
            else:
                q, r = 0.0, 0.0

            if m >= q and m >= r:
                nms[y, x] = m
            else:
                nms[y, x] = 0.0

    return nms

@numba.jit(nopython=True, fastmath=True)
def _hysteresis_thresholding(nms_img: np.ndarray, low_thresh: float, high_thresh: float) -> np.ndarray:
    '''
    Perform hysteresis thresholding to link edges.

    Args:
        nms_img (np.ndarray): Non-maximum suppressed image.
        low_thresh (float): Low threshold for weak edges.
        high_thresh (float): High threshold for strong edges.

    Returns:
        np.ndarray: Binary edge map of shape (H, W) with values 0 or 255.
    '''
    H, W = nms_img.shape
    res = np.zeros((H, W), dtype=np.uint8)

    STRONG = 255
    WEAK = 128

    queue_x = []
    queue_y = []

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            val = nms_img[y, x]
            if val >= high_thresh:
                res[y, x] = STRONG
                queue_x.append(x)
                queue_y.append(y)
            elif val >= low_thresh:
                res[y, x] = WEAK

    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    head = 0
    while head < len(queue_x):
        cx = queue_x[head]
        cy = queue_y[head]
        head += 1

        for i in range(8):
            nx = cx + dx[i]
            ny = cy + dy[i]
            if 0 <= nx < W and 0 <= ny < H:
                if res[ny, nx] == WEAK:
                    res[ny, nx] = STRONG
                    queue_x.append(nx)
                    queue_y.append(ny)

    for y in range(H):
        for x in range(W):
            if res[y, x] != STRONG:
                res[y, x] = 0

    return res

def apply_canny(
    image: Union[np.ndarray, torch.Tensor],
    low_thresh: float = 50.0,
    high_thresh: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply the Canny edge detection algorithm to an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        low_thresh (float): Low threshold for hysteresis.
        high_thresh (float): High threshold for hysteresis.
        kernel_size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation of Gaussian kernel.
        device (str): Device to perform PyTorch computations on.

    Returns:
        np.ndarray: Detected binary edge map.
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    mag, angles = preprocess_canny_pytorch(img_tensor, kernel_size=kernel_size, sigma=sigma, device=device)
    nms_img = _non_max_suppression(mag, angles)
    edge_map = _hysteresis_thresholding(nms_img, low_thresh=low_thresh, high_thresh=high_thresh)

    return edge_map
