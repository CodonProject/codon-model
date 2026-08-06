from codon import *
from codon.ops.math import l2_hys_normalize


def preprocess_hog_pytorch(
    img_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Preprocess image tensor with PyTorch to compute gradient magnitude and angle.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W) or (1, C, H, W).
        device (str): Device to perform PyTorch calculations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradient magnitude and angle arrays.
    '''
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    img_tensor = img_tensor.to(device).float()
    C = img_tensor.shape[1]

    kernel_x = torch.tensor([[[-1., 0., 1.]]], device=device).repeat(C, 1, 1, 1)
    kernel_y = torch.tensor([[[-1.], [0.], [1.]]], device=device).repeat(C, 1, 1, 1)

    gx = F.conv2d(img_tensor, kernel_x, padding=(0, 1), groups=C)
    gy = F.conv2d(img_tensor, kernel_y, padding=(1, 0), groups=C)

    mag = torch.sqrt(gx**2 + gy**2)

    if C > 1:
        max_mag, max_idx = torch.max(mag, dim=1, keepdim=True)
        gx = torch.gather(gx, dim=1, index=max_idx).squeeze()
        gy = torch.gather(gy, dim=1, index=max_idx).squeeze()
        mag = max_mag.squeeze()
    else:
        gx = gx.squeeze()
        gy = gy.squeeze()
        mag = mag.squeeze()

    angle = torch.atan2(gy, gx) % np.pi

    return mag.cpu().numpy(), angle.cpu().numpy()

@numba.jit(nopython=True, fastmath=True)
def _compute_cell_histograms(
    mag: np.ndarray,
    angle: np.ndarray,
    cell_h: int,
    cell_w: int,
    n_cells_y: int,
    n_cells_x: int,
    orientations: int = 9
) -> np.ndarray:
    '''
    Compute orientation histograms for each cell.

    Args:
        mag (np.ndarray): Gradient magnitude array.
        angle (np.ndarray): Gradient angle array.
        cell_h (int): Cell height in pixels.
        cell_w (int): Cell width in pixels.
        n_cells_y (int): Number of cells along Y axis.
        n_cells_x (int): Number of cells along X axis.
        orientations (int): Number of orientation bins.

    Returns:
        np.ndarray: Cell histograms array of shape (n_cells_y, n_cells_x, orientations).
    '''
    histograms = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_width = np.pi / orientations

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            
            y_start = cy * cell_h
            x_start = cx * cell_w
            
            for y in range(y_start, y_start + cell_h):
                for x in range(x_start, x_start + cell_w):
                    m = mag[y, x]
                    a = angle[y, x]

                    b = a / bin_width
                    b0 = int(np.floor(b)) % orientations
                    b1 = (b0 + 1) % orientations

                    w1 = b - np.floor(b)
                    w0 = 1.0 - w1

                    histograms[cy, cx, b0] += m * w0
                    histograms[cy, cx, b1] += m * w1

    return histograms

@numba.jit(nopython=True, fastmath=True)
def _normalize_blocks(
    histograms: np.ndarray,
    block_h: int,
    block_w: int,
    eps: float = 1e-5
) -> np.ndarray:
    '''
    Normalize cell histograms within overlapping blocks (L2-Hys normalization).

    Args:
        histograms (np.ndarray): Cell histograms.
        block_h (int): Block height in number of cells.
        block_w (int): Block width in number of cells.
        eps (float): Small constant for numerical stability.

    Returns:
        np.ndarray: Flattened HOG feature descriptor.
    '''
    n_cells_y, n_cells_x, orientations = histograms.shape
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    block_feat_dim = block_h * block_w * orientations
    blocks = np.zeros((n_blocks_y, n_blocks_x, block_feat_dim), dtype=np.float32)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block_vec = histograms[by : by + block_h, bx : bx + block_w, :].ravel()
            blocks[by, bx, :] = l2_hys_normalize(block_vec, eps=eps, max_val=0.2)

    return blocks.ravel()


def _visualize_hog(
    histograms: np.ndarray,
    cell_h: int,
    cell_w: int,
    orientations: int = 9
) -> np.ndarray:
    '''
    Create a visual representation image of HOG histograms.

    Args:
        histograms (np.ndarray): Cell histograms.
        cell_h (int): Cell height in pixels.
        cell_w (int): Cell width in pixels.
        orientations (int): Number of orientation bins.

    Returns:
        np.ndarray: Visualization image of HOG features.
    '''
    n_cells_y, n_cells_x, _ = histograms.shape
    hog_img = np.zeros((n_cells_y * cell_h, n_cells_x * cell_w), dtype=np.float32)
    
    bin_width = np.pi / orientations
    mid_angles = (np.arange(orientations) + 0.5) * bin_width

    radius = min(cell_h, cell_w) // 2 - 1

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            center_y = cy * cell_h + cell_h // 2
            center_x = cx * cell_w + cell_w // 2

            hist = histograms[cy, cx, :]
            max_val = np.max(hist) + 1e-5

            for b in range(orientations):
                magnitude = hist[b] / max_val
                if magnitude < 0.1:
                    continue

                ang = mid_angles[b]
                dx = int(radius * np.cos(ang + np.pi / 2.0))
                dy = int(radius * np.sin(ang + np.pi / 2.0))

                steps = max(abs(dx), abs(dy), 1)
                for i in range(-steps, steps + 1):
                    x = center_x + int(i * dx / steps)
                    y = center_y + int(i * dy / steps)
                    if 0 <= y < hog_img.shape[0] and 0 <= x < hog_img.shape[1]:
                        hog_img[y, x] = max(hog_img[y, x], magnitude)

    return hog_img

def apply_hog(
    image: Union[np.ndarray, torch.Tensor],
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False,
    device: str = 'cpu'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    Extract Histogram of Oriented Gradients (HOG) features from an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        orientations (int): Number of orientation bins.
        pixels_per_cell (Tuple[int, int]): Size of a cell in pixels (height, width).
        cells_per_block (Tuple[int, int]): Number of cells in each block (height, width).
        visualize (bool): Whether to return a HOG visualization image.
        device (str): Device to use for PyTorch operations.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            HOG descriptor vector, and optionally HOG visualization image.
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block

    mag, angle = preprocess_hog_pytorch(img_tensor, device=device)

    H, W = mag.shape
    n_cells_y = H // cell_h
    n_cells_x = W // cell_w

    assert n_cells_y >= block_h and n_cells_x >= block_w, \
        f"图像尺寸 ({H}x{W}) 对于给定的 Cell ({pixels_per_cell}) 和 Block ({cells_per_block}) 太小！"

    histograms = _compute_cell_histograms(mag, angle, cell_h, cell_w, n_cells_y, n_cells_x, orientations=orientations)
    descriptor = _normalize_blocks(histograms, block_h, block_w)

    if visualize:
        hog_image = _visualize_hog(histograms, cell_h, cell_w, orientations=orientations)
        return descriptor, hog_image

    return descriptor
