from codon import *


def preprocess_hough_pytorch(
    img_tensor: torch.Tensor,
    edge_threshold: float = 128.0,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Extract edge coordinate indices from an image tensor.

    Args:
        img_tensor (torch.Tensor): Input edge image tensor of shape (H, W) or (1, H, W).
        edge_threshold (float): Threshold to consider a pixel as an edge.
        device (str): Device to perform computations.

    Returns:
        np.ndarray: Array of shape (N, 2) containing edge coordinate indices (y, x).
    '''
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze()

    img_tensor = img_tensor.to(device).float()
    
    edge_indices = torch.nonzero(img_tensor >= edge_threshold)
    
    return edge_indices.cpu().numpy()

@numba.jit(nopython=True, fastmath=True)
def _hough_accumulate(
    edge_coords: np.ndarray,
    num_rhos: int,
    num_thetas: int,
    sin_t: np.ndarray,
    cos_t: np.ndarray,
    max_dist: float,
    rho_res: float
) -> np.ndarray:
    '''
    Accumulate votes in the Hough parameter space.

    Args:
        edge_coords (np.ndarray): Edge coordinates.
        num_rhos (int): Number of rho bins.
        num_thetas (int): Number of theta bins.
        sin_t (np.ndarray): Sine of the theta bins.
        cos_t (np.ndarray): Cosine of the theta bins.
        max_dist (float): Maximum possible distance from origin (diagonal of image).
        rho_res (float): Resolution of the distance parameter.

    Returns:
        np.ndarray: Accumulator array of shape (num_rhos, num_thetas).
    '''
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    n_points = edge_coords.shape[0]

    for i in range(n_points):
        y = edge_coords[i, 0]
        x = edge_coords[i, 1]

        for t_idx in range(num_thetas):
            rho = x * cos_t[t_idx] + y * sin_t[t_idx]
            
            rho_idx = int(np.round((rho + max_dist) / rho_res))
            
            if 0 <= rho_idx < num_rhos:
                accumulator[rho_idx, t_idx] += 1

    return accumulator

@numba.jit(nopython=True, fastmath=True)
def _find_hough_peaks(
    accumulator: np.ndarray,
    threshold: int,
    nhood_r: int = 5,
    nhood_t: int = 5
) -> List[Tuple[int, int, int]]:
    '''
    Find peaks in the accumulator array using local non-maximum suppression.

    Args:
        accumulator (np.ndarray): Hough accumulator array.
        threshold (int): Minimum vote count to consider a peak.
        nhood_r (int): Neighborhood size in the rho dimension.
        nhood_t (int): Neighborhood size in the theta dimension.

    Returns:
        List[Tuple[int, int, int]]: List of (rho_idx, theta_idx, votes) tuples representing peaks.
    '''
    num_rhos, num_thetas = accumulator.shape
    peaks = []

    for r in range(num_rhos):
        for t in range(num_thetas):
            val = accumulator[r, t]
            if val < threshold:
                continue

            is_max = True
            for dr in range(-nhood_r, nhood_r + 1):
                for dt in range(-nhood_t, nhood_t + 1):
                    if dr == 0 and dt == 0:
                        continue
                    
                    nr = r + dr
                    nt = (t + dt) % num_thetas
                    
                    if 0 <= nr < num_rhos:
                        if accumulator[nr, nt] > val:
                            is_max = False
                            break
                if not is_max:
                    break

            if is_max:
                peaks.append((r, t, val))

    return peaks

def hough_lines_to_endpoints(lines: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    '''
    Convert Hough parameters (rho, theta) to segment endpoints.

    Args:
        lines (np.ndarray): Detected lines array of shape (N, 2).
        img_shape (Tuple[int, int]): Dimensions of the image (H, W).

    Returns:
        np.ndarray: Line segment endpoints array of shape (N, 4) containing [x1, y1, x2, y2].
    '''
    H, W = img_shape[:2]
    endpoints = []

    for rho, theta in lines:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        points = []

        if abs(sin_t) > 1e-5:
            y = rho / sin_t
            if 0 <= y < H:
                points.append((0, y))

        if abs(sin_t) > 1e-5:
            y = (rho - (W - 1) * cos_t) / sin_t
            if 0 <= y < H:
                points.append((W - 1, y))

        if abs(cos_t) > 1e-5:
            x = rho / cos_t
            if 0 <= x < W:
                points.append((x, 0))

        if abs(cos_t) > 1e-5:
            x = (rho - (H - 1) * sin_t) / cos_t
            if 0 <= x < W:
                points.append((x, H - 1))

        unique_pts = []
        for pt in points:
            if not any(np.isclose(pt, u, atol=1e-3).all() for u in unique_pts):
                unique_pts.append(pt)

        if len(unique_pts) >= 2:
            x1, y1 = unique_pts[0]
            x2, y2 = unique_pts[1]
            endpoints.append([x1, y1, x2, y2])

    return np.array(endpoints)

def apply_hough(
    image: Union[np.ndarray, torch.Tensor],
    rho_res: float = 1.0,
    theta_res: float = np.pi/180.0,
    threshold: int = 50,
    edge_threshold: float = 128.0,
    return_endpoints: bool = False,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply the Hough Transform for detecting lines in an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        rho_res (float): Resolution of the distance parameter in pixels.
        theta_res (float): Resolution of the angle parameter in radians.
        threshold (int): Minimum vote count to register a line.
        edge_threshold (float): Threshold to classify a pixel as an edge.
        return_endpoints (bool): Whether to return endpoint coords [x1, y1, x2, y2].
        device (str): Device to use for PyTorch operations.

    Returns:
        np.ndarray: Hough parameters (rho, theta) or segment endpoints (x1, y1, x2, y2).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    H, W = img_tensor.shape[-2:]

    edge_coords = preprocess_hough_pytorch(img_tensor, edge_threshold=edge_threshold, device=device)

    if len(edge_coords) == 0:
        return np.empty((0, 4 if return_endpoints else 2))

    max_dist = np.sqrt(H**2 + W**2)
    num_rhos = int(np.ceil(2 * max_dist / rho_res)) + 1
    
    thetas = np.arange(0, np.pi, theta_res)
    num_thetas = len(thetas)

    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)

    accumulator = _hough_accumulate(edge_coords, num_rhos, num_thetas, sin_t, cos_t, max_dist, rho_res)

    peaks = _find_hough_peaks(accumulator, threshold=threshold, nhood_r=5, nhood_t=5)

    if len(peaks) == 0:
        return np.empty((0, 4 if return_endpoints else 2))

    lines = []
    for r_idx, t_idx, votes in peaks:
        rho = (r_idx * rho_res) - max_dist
        theta = thetas[t_idx]
        lines.append([rho, theta])

    lines = np.array(lines)

    if return_endpoints:
        return hough_lines_to_endpoints(lines, (H, W))

    return lines
