from codon import *
from codon.ops import prepare_input_tensor, rgb_to_lab

def preprocess_slic_pytorch(
    img_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Preprocess image tensor by converting to LAB and calculating gradient.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (1, H, W) or (3, H, W).
        device (str): Device to perform computations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: LAB image array and gradient map.
    '''
    img_tensor = prepare_input_tensor(img_tensor, device=device)
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)

    lab_tensor = rgb_to_lab(img_tensor, device=device)

    L_chan = lab_tensor[0:1].unsqueeze(0)
    sobel_x = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], device=device).view(1, 1, 3, 3)
    
    gx = F.conv2d(L_chan, sobel_x, padding=1).squeeze()
    gy = F.conv2d(L_chan, sobel_y, padding=1).squeeze()
    grad = torch.sqrt(gx**2 + gy**2)

    return lab_tensor.permute(1, 2, 0).cpu().numpy(), grad.cpu().numpy()

@numba.jit(nopython=True, fastmath=True)
def _init_centers(lab_img: np.ndarray, grad: np.ndarray, n_segments: int) -> Tuple[np.ndarray, int]:
    '''
    Initialize superpixel cluster centers at locations of minimum gradient.

    Args:
        lab_img (np.ndarray): LAB image array of shape (H, W, 3).
        grad (np.ndarray): Image gradient magnitude map of shape (H, W).
        n_segments (int): Approximate number of target superpixels.

    Returns:
        Tuple[np.ndarray, int]: Array of cluster centers and grid step size S.
    '''
    H, W, _ = lab_img.shape
    S = int(np.sqrt((H * W) / n_segments))

    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            min_g = 1e9
            best_x, best_y = x, y
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if grad[ny, nx] < min_g:
                            min_g = grad[ny, nx]
                            best_x, best_y = nx, ny
            
            l, a, b = lab_img[best_y, best_x]
            centers.append([l, a, b, float(best_x), float(best_y)])

    return np.array(centers, dtype=np.float32), S

@numba.jit(nopython=True, fastmath=True)
def _slic_cluster(
    lab_img: np.ndarray,
    centers: np.ndarray,
    S: int,
    compactness: float = 10.0,
    max_iter: int = 10
) -> np.ndarray:
    '''
    Perform SLIC clustering using local k-means optimization.

    Args:
        lab_img (np.ndarray): LAB image array of shape (H, W, 3).
        centers (np.ndarray): Initial cluster centers.
        S (int): Grid step size.
        compactness (float): Parameter balancing color similarity and spatial proximity.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Label map of shape (H, W).
    '''
    H, W, _ = lab_img.shape
    K = len(centers)
    labels = -np.ones((H, W), dtype=numba.int32)
    distances = np.full((H, W), 1e9, dtype=numba.float32)

    inv_spatial_scale = (compactness / S) ** 2

    for iteration in range(max_iter):
        distances.fill(1e9)

        for k in range(K):
            l_c, a_c, b_c, x_c, y_c = centers[k]

            y_min = max(0, int(y_c - S))
            y_max = min(H, int(y_c + S + 1))
            x_min = max(0, int(x_c - S))
            x_max = min(W, int(x_c + S + 1))

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    l, a, b = lab_img[y, x]

                    d_color = (l - l_c)**2 + (a - a_c)**2 + (b - b_c)**2
                    d_spatial = (x - x_c)**2 + (y - y_c)**2
                    
                    D_sq = d_color + d_spatial * inv_spatial_scale

                    if D_sq < distances[y, x]:
                        distances[y, x] = D_sq
                        labels[y, x] = k

        centers_new = np.zeros_like(centers)
        counts = np.zeros(K, dtype=np.float32)

        for y in range(H):
            for x in range(W):
                k = labels[y, x]
                if k >= 0:
                    l, a, b = lab_img[y, x]
                    centers_new[k, 0] += l
                    centers_new[k, 1] += a
                    centers_new[k, 2] += b
                    centers_new[k, 3] += x
                    centers_new[k, 4] += y
                    counts[k] += 1.0

        for k in range(K):
            if counts[k] > 0:
                centers_new[k] /= counts[k]
            else:
                centers_new[k] = centers[k]

        centers = centers_new

    return labels

@numba.jit(nopython=True, fastmath=True)
def _enforce_connectivity(labels: np.ndarray, min_element_size: int) -> np.ndarray:
    '''
    Enforce spatial connectivity of superpixel labels to remove small orphan segments.

    Args:
        labels (np.ndarray): Label map from clustering.
        min_element_size (int): Minimum pixel count for a superpixel.

    Returns:
        np.ndarray: Connected label map of shape (H, W).
    '''
    H, W = labels.shape
    new_labels = -np.ones((H, W), dtype=numba.int32)
    
    current_label = 0
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    for y in range(H):
        for x in range(W):
            if new_labels[y, x] < 0:
                old_label = labels[y, x]
                
                segment = [(x, y)]
                head = 0
                adj_label = current_label

                new_labels[y, x] = current_label

                while head < len(segment):
                    cx, cy = segment[head]
                    head += 1

                    for i in range(4):
                        nx, ny = cx + dx[i], cy + dy[i]
                        if 0 <= nx < W and 0 <= ny < H:
                            if new_labels[ny, nx] < 0 and labels[ny, nx] == old_label:
                                new_labels[ny, nx] = current_label
                                segment.append((nx, ny))
                            elif new_labels[ny, nx] >= 0 and new_labels[ny, nx] != current_label:
                                adj_label = new_labels[ny, nx]

                if len(segment) < min_element_size:
                    for px, py in segment:
                        new_labels[py, px] = adj_label
                else:
                    current_label += 1

        return new_labels

def apply_slic(
    image: Union[np.ndarray, torch.Tensor],
    n_segments: int = 100,
    compactness: float = 10.0,
    max_iter: int = 10,
    enforce_connectivity: bool = True,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply Simple Linear Iterative Clustering (SLIC) to an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        n_segments (int): Number of target superpixels.
        compactness (float): Compactness parameter for superpixels.
        max_iter (int): Maximum number of iterations.
        enforce_connectivity (bool): Whether to enforce connected segments.
        device (str): Device to use for PyTorch operations.

    Returns:
        np.ndarray: Label map of shape (H, W).
    '''
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image
        if img_tensor.ndim == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

    lab_img, grad = preprocess_slic_pytorch(img_tensor, device=device)

    centers, S = _init_centers(lab_img, grad, n_segments)

    labels = _slic_cluster(lab_img, centers, S, compactness=compactness, max_iter=max_iter)

    if enforce_connectivity:
        min_element_size = (S * S) // 4
        labels = _enforce_connectivity(labels, min_element_size)

    return labels

def find_boundaries(labels: np.ndarray) -> np.ndarray:
    '''
    Find boundaries between different superpixel label regions.

    Args:
        labels (np.ndarray): Superpixel label map of shape (H, W).

    Returns:
        np.ndarray: Boolean boundary map of shape (H, W).
    '''
    H, W = labels.shape
    boundaries = np.zeros((H, W), dtype=bool)

    boundaries[:-1, :] |= (labels[:-1, :] != labels[1:, :])
    boundaries[:, :-1] |= (labels[:, :-1] != labels[:, 1:])

    return boundaries
