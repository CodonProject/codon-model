from codon import *
from codon.ops import compute_image_gradients, angle_diff


def preprocess_image_pytorch(
    img_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Preprocess input image to compute gradient magnitude, angles, and sorted indices.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W) or (1, C, H, W).
        device (str): Device to perform processing on.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Gradient magnitude array.
            - Gradient angle array.
            - Sorted gradient magnitude indices in descending order.
    '''
    mag, _, _, angles = compute_image_gradients(img_tensor, blur_sigma=0.8, kernel_size=5, device=device)
  
    flat_mag = mag.view(-1)
    sorted_indices = torch.argsort(flat_mag, descending=True)
  
    return mag.cpu().numpy(), angles.cpu().numpy(), sorted_indices.cpu().numpy()

@numba.jit(nopython=True, fastmath=True)
def _angle_diff(a1: float, a2: float) -> float:
    '''
    Calculate the absolute difference between two angles.

    Args:
        a1 (float): First angle in radians.
        a2 (float): Second angle in radians.

    Returns:
        float: Absolute angle difference in radians.
    '''
    diff = np.abs(a1 - a2)
    if diff > np.pi:
        diff = 2.0 * np.pi - diff
    if diff > np.pi / 2.0:
        diff = np.pi - diff
    return diff

@numba.jit(nopython=True, fastmath=True)
def _grow_region(
    mag: np.ndarray,
    angles: np.ndarray,
    used: np.ndarray,
    seed_x: int,
    seed_y: int,
    H: int,
    W: int,
    ang_thresh: float = 0.3926
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Grow a region starting from a seed point based on angle similarity.

    Args:
        mag (np.ndarray): Gradient magnitude array.
        angles (np.ndarray): Gradient angle array.
        used (np.ndarray): Boolean mask of visited coordinates.
        seed_x (int): X coordinate of the seed point.
        seed_y (int): Y coordinate of the seed point.
        H (int): Image height.
        W (int): Image width.
        ang_thresh (float): Maximum angle difference threshold.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinates of the grown region.
    '''
    region_x = [seed_x]
    region_y = [seed_y]
    
    seed_angle = angles[seed_y, seed_x]
    used[seed_y, seed_x] = True
    
    head = 0
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    
    sum_angle_x = np.cos(seed_angle)
    sum_angle_y = np.sin(seed_angle)
    
    while head < len(region_x):
        cx = region_x[head]
        cy = region_y[head]
        head += 1
        
        mean_angle = np.arctan2(sum_angle_y, sum_angle_x)
        
        for i in range(8):
            nx, ny = cx + dx[i], cy + dy[i]
            if 0 <= nx < W and 0 <= ny < H:
                if not used[ny, nx]:
                    if angle_diff(angles[ny, nx], mean_angle) < ang_thresh:
                        used[ny, nx] = True
                        region_x.append(nx)
                        region_y.append(ny)
                        sum_angle_x += np.cos(angles[ny, nx])
                        sum_angle_y += np.sin(angles[ny, nx])
                        
    return np.array(region_x), np.array(region_y)

@numba.jit(nopython=True, fastmath=True)
def _fit_rectangle(pts_x: np.ndarray, pts_y: np.ndarray) -> np.ndarray:
    '''
    Fit a minimum bounding rectangle to a set of points.

    Args:
        pts_x (np.ndarray): X coordinates of the points.
        pts_y (np.ndarray): Y coordinates of the points.

    Returns:
        np.ndarray: Array containing [x1, y1, x2, y2, width].
    '''
    n = len(pts_x)
    if n < 5:
        return np.zeros(5)
        
    cx = np.mean(pts_x)
    cy = np.mean(pts_y)
    
    vx = pts_x - cx
    vy = pts_y - cy
    
    mxx = np.sum(vx * vx) / n
    myy = np.sum(vy * vy) / n
    mxy = np.sum(vx * vy) / n
    
    theta = 0.5 * np.arctan2(2.0 * mxy, mxx - myy) + np.pi / 2.0
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    proj1 = vx * cos_t + vy * sin_t
    proj2 = -vx * sin_t + vy * cos_t
    
    l_min, l_max = np.min(proj1), np.max(proj1)
    w_min, w_max = np.min(proj2), np.max(proj2)
    
    x1 = cx + l_min * cos_t
    y1 = cy + l_min * sin_t
    x2 = cx + l_max * cos_t
    y2 = cy + l_max * sin_t
    width = w_max - w_min
    
    return np.array([x1, y1, x2, y2, width])

@numba.jit(nopython=True, fastmath=True)
def lsd_core_numba(
    mag: np.ndarray,
    angles: np.ndarray,
    sorted_indices: np.ndarray,
    min_length: float = 15.0,
    grad_thresh: float = 20.0
) -> List[np.ndarray]:
    '''
    Core line segment detection algorithm optimized with Numba.

    Args:
        mag (np.ndarray): Gradient magnitude array.
        angles (np.ndarray): Gradient angle array.
        sorted_indices (np.ndarray): Sorted indices of gradient magnitudes.
        min_length (float): Minimum length of line segments.
        grad_thresh (float): Minimum gradient threshold.

    Returns:
        List[np.ndarray]: List of detected line segments [x1, y1, x2, y2, width].
    '''
    H, W = mag.shape
    used = np.zeros((H, W), dtype=numba.boolean)
    lines = []
    
    for idx in sorted_indices:
        y = idx // W
        x = idx % W
        
        if used[y, x] or mag[y, x] < grad_thresh:
            continue
            
        pts_x, pts_y = _grow_region(mag, angles, used, x, y, H, W)
        
        if len(pts_x) < min_length:
            continue
            
        rect = _fit_rectangle(pts_x, pts_y)
        dx = rect[2] - rect[0]
        dy = rect[3] - rect[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length >= min_length:
            lines.append(rect)
            
    return lines

@numba.jit(nopython=True, fastmath=True)
def _fit_ellipse_direct(pts_x: np.ndarray, pts_y: np.ndarray) -> np.ndarray:
    '''
    Directly fit an ellipse to a set of points using algebraic distance minimization.

    Args:
        pts_x (np.ndarray): X coordinates of points.
        pts_y (np.ndarray): Y coordinates of points.

    Returns:
        np.ndarray: Ellipse parameters [cx, cy, axis_a, axis_b, phi].
    '''
    n = len(pts_x)
    if n < 6:
        return np.zeros(5)

    X = pts_x.astype(numba.float64)
    Y = pts_y.astype(numba.float64)
    
    D1 = np.vstack((X*X, X*Y, Y*Y)).T
    D2 = np.vstack((X, Y, np.ones(n))).T
    
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    
    if np.linalg.det(S3) == 0:
        return np.zeros(5)
        
    T = -np.dot(np.linalg.inv(S3), S2.T)
    M = S1 + np.dot(S2, T)
    
    C = np.zeros((3, 3))
    C[0, 2] = 2.0
    C[1, 1] = -1.0
    C[2, 0] = 2.0
    
    invC_M = np.dot(np.linalg.inv(C), M)
    evals, evecs = np.linalg.eig(invC_M)
    
    cond = 4.0 * evecs[0, :] * evecs[2, :] - evecs[1, :]**2
    valid_idx = -1
    for i in range(3):
        if cond[i] > 0:
            valid_idx = i
            break
            
    if valid_idx == -1:
        return np.zeros(5)
        
    a1 = evecs[:, valid_idx]
    a2 = np.dot(T, a1)
    
    A, B, C_coef, D_coef, E_coef, F_coef = a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]
    
    num = 2 * (A * E_coef**2 + C_coef * D_coef**2 - B * D_coef * E_coef + (B**2 - 4 * A * C_coef) * F_coef)
    den1 = (B**2 - 4 * A * C_coef) * (np.sqrt((A - C_coef)**2 + B**2) - (A + C_coef))
    den2 = (B**2 - 4 * A * C_coef) * (-np.sqrt((A - C_coef)**2 + B**2) - (A + C_coef))
    
    if den1 <= 0 or den2 <= 0:
        return np.zeros(5)
        
    cx = (2 * C_coef * D_coef - B * E_coef) / (B**2 - 4 * A * C_coef)
    cy = (2 * A * E_coef - B * D_coef) / (B**2 - 4 * A * C_coef)
    axis_a = np.sqrt(abs(num / den1))
    axis_b = np.sqrt(abs(num / den2))
    phi = 0.5 * np.arctan2(B, A - C_coef)
    
    return np.array([cx, cy, axis_a, axis_b, phi])

@numba.jit(nopython=True, fastmath=True)
def elsd_core_numba(
    mag: np.ndarray,
    angles: np.ndarray,
    sorted_indices: np.ndarray,
    min_arc_len: float = 20.0,
    grad_thresh: float = 20.0
) -> List[np.ndarray]:
    '''
    Core ellipse segment detection algorithm optimized with Numba.

    Args:
        mag (np.ndarray): Gradient magnitude array.
        angles (np.ndarray): Gradient angle array.
        sorted_indices (np.ndarray): Sorted indices of gradient magnitudes.
        min_arc_len (float): Minimum arc length of the ellipse segments.
        grad_thresh (float): Minimum gradient threshold.

    Returns:
        List[np.ndarray]: List of detected ellipse parameters.
    '''
    H, W = mag.shape
    used = np.zeros((H, W), dtype=numba.boolean)
    ellipses = []
    
    for idx in sorted_indices:
        y = idx // W
        x = idx % W
        
        if used[y, x] or mag[y, x] < grad_thresh:
            continue
            
        pts_x, pts_y = _grow_region(mag, angles, used, x, y, H, W, ang_thresh=0.7854)
        
        if len(pts_x) < min_arc_len:
            continue
            
        ellipse = _fit_ellipse_direct(pts_x, pts_y)
        
        if ellipse[2] > 0 and ellipse[3] > 0:
            aspect_ratio = max(ellipse[2], ellipse[3]) / min(ellipse[2], ellipse[3])
            if aspect_ratio < 10.0 and ellipse[2] < max(H, W) and ellipse[3] < max(H, W):
                ellipses.append(ellipse)
                
    return ellipses

def apply_lsd(
    image: Union[np.ndarray, torch.Tensor],
    min_length: float = 15.0,
    grad_thresh: float = 20.0,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply the Line Segment Detector (LSD) algorithm to an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        min_length (float): Minimum line segment length.
        grad_thresh (float): Minimum gradient threshold.
        device (str): Device to use for PyTorch operations.

    Returns:
        np.ndarray: Detected line segments as a NumPy array of shape (N, 5).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    mag, angles, sorted_indices = preprocess_image_pytorch(img_tensor, device=device)
    lines = lsd_core_numba(mag, angles, sorted_indices, min_length=min_length, grad_thresh=grad_thresh)
    
    if len(lines) == 0:
        return np.empty((0, 5))
    return np.array(lines)

def apply_elsd(
    image: Union[np.ndarray, torch.Tensor],
    min_arc_len: float = 20.0,
    grad_thresh: float = 20.0,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply the Ellipse Line Segment Detector (ELSD) algorithm to an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        min_arc_len (float): Minimum arc length for detected ellipse segments.
        grad_thresh (float): Minimum gradient threshold.
        device (str): Device to use for PyTorch operations.

    Returns:
        np.ndarray: Detected ellipse parameters as a NumPy array of shape (N, 5).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    mag, angles, sorted_indices = preprocess_image_pytorch(img_tensor, device=device)
    ellipses = elsd_core_numba(mag, angles, sorted_indices, min_arc_len=min_arc_len, grad_thresh=grad_thresh)
    
    if len(ellipses) == 0:
        return np.empty((0, 5))
    return np.array(ellipses)
