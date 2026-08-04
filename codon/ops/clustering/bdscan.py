from codon import *


def _rgb_to_lab_pytorch(img_tensor: torch.Tensor) -> torch.Tensor:
    '''
    Convert an RGB or sRGB image tensor to CIELAB color space using PyTorch.

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (3, H, W) or (1, 3, H, W).

    Returns:
        torch.Tensor: CIELAB image tensor of shape (3, H, W).
    '''
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

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

    lab = torch.cat([L, a, b], dim=-1).squeeze(0).permute(2, 0, 1)
    return lab

def preprocess_dbscan_pytorch(
    img_tensor: torch.Tensor,
    use_lab: bool = True,
    spatial_weight: float = 0.0,
    device: str = 'cpu'
) -> Tuple[np.ndarray, int, int]:
    '''
    Preprocess image tensor to generate a flattened feature matrix for DBSCAN.

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (H, W), (C, H, W) or (1, C, H, W).
        use_lab (bool): Whether to convert RGB features to LAB space.
        spatial_weight (float): Multiplier for spatial coords (y, x) feature dimensions.
        device (str): Device to use for PyTorch operations.

    Returns:
        Tuple[np.ndarray, int, int]: Flattened feature matrix of shape (N, D), H, and W.
    '''
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
        
    img_tensor = img_tensor.to(device).float()
    if img_tensor.dim() == 3 and img_tensor.shape[2] in [1, 3]:
        img_tensor = img_tensor.permute(2, 0, 1)

    C, H, W = img_tensor.shape

    if C == 3 and use_lab:
        color_feat = _rgb_to_lab_pytorch(img_tensor)
    else:
        color_feat = img_tensor

    feat_list = [color_feat.permute(1, 2, 0)]

    if spatial_weight > 0.0:
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        y_feat = (y_grid / max(H, W)) * 100.0 * spatial_weight
        x_feat = (x_grid / max(H, W)) * 100.0 * spatial_weight
        spatial_feat = torch.stack([y_feat, x_feat], dim=-1)
        feat_list.append(spatial_feat)

    full_feat = torch.cat(feat_list, dim=-1)
    N, D = H * W, full_feat.shape[-1]
    data_flat = full_feat.view(N, D)

    return data_flat.cpu().numpy(), H, W

@numba.jit(nopython=True, fastmath=True)
def _region_query(
    data: np.ndarray,
    point_idx: int,
    eps_sq: float,
    spatial_h: int,
    spatial_w: int,
    spatial_weight: float,
    max_spatial_diff: int,
    neighbor_buf: np.ndarray
) -> int:
    '''
    Query the neighbors of a point within epsilon distance.

    Args:
        data (np.ndarray): Flattened feature matrix of shape (N, D).
        point_idx (int): Index of the reference query point.
        eps_sq (float): Squared epsilon search distance threshold.
        spatial_h (int): Spatial height limit.
        spatial_w (int): Spatial width limit.
        spatial_weight (float): Weight of spatial coordinate features.
        max_spatial_diff (int): Maximum spatial window boundary.
        neighbor_buf (np.ndarray): Buffer array to store indexes of found neighbors.

    Returns:
        int: Number of neighbors found.
    '''
    N, D = data.shape
    count = 0

    if spatial_h > 0 and spatial_w > 0 and spatial_weight > 0.0 and max_spatial_diff > 0:
        yi = point_idx // spatial_w
        xi = point_idx % spatial_w
        
        y_min = max(0, yi - max_spatial_diff)
        y_max = min(spatial_h, yi + max_spatial_diff + 1)
        x_min = max(0, xi - max_spatial_diff)
        x_max = min(spatial_w, xi + max_spatial_diff + 1)

        for ny in range(y_min, y_max):
            for nx in range(x_min, x_max):
                j = ny * spatial_w + nx
                d2 = 0.0
                for d in range(D):
                    diff = data[point_idx, d] - data[j, d]
                    d2 += diff * diff
                    if d2 > eps_sq:
                        break
                if d2 <= eps_sq:
                    neighbor_buf[count] = j
                    count += 1
    else:
        for j in range(N):
            d2 = 0.0
            for d in range(D):
                diff = data[point_idx, d] - data[j, d]
                d2 += diff * diff
                if d2 > eps_sq:
                    break
            if d2 <= eps_sq:
                neighbor_buf[count] = j
                count += 1

    return count

@numba.jit(nopython=True, fastmath=True)
def _dbscan_numba(
    data: np.ndarray,
    eps: float,
    min_samples: int,
    spatial_h: int = 0,
    spatial_w: int = 0,
    spatial_weight: float = 0.0
) -> np.ndarray:
    '''
    Execute DBSCAN clustering algorithm using Numba acceleration.

    Args:
        data (np.ndarray): Flattened feature matrix of shape (N, D).
        eps (float): Epsilon neighborhood search distance.
        min_samples (int): Minimum number of neighbor samples to classify as core points.
        spatial_h (int): Height of the image grid if spatial clustering.
        spatial_w (int): Width of the image grid if spatial clustering.
        spatial_weight (float): Multiplier weight for spatial features.

    Returns:
        np.ndarray: Assigned cluster labels array of shape (N,).
    '''
    N, _ = data.shape
    eps_sq = eps * eps
    
    UNVISITED = -100
    NOISE = -1

    labels = np.full(N, UNVISITED, dtype=np.int32)

    max_spatial_diff = 0
    if spatial_h > 0 and spatial_w > 0 and spatial_weight > 0.0:
        scale = (100.0 * spatial_weight) / max(spatial_h, spatial_w)
        if scale > 0:
            max_spatial_diff = int(np.ceil(eps / scale)) + 1

    neighbor_buf = np.zeros(N, dtype=np.int32)
    sub_neighbor_buf = np.zeros(N, dtype=np.int32)
    queue = np.zeros(N, dtype=np.int32)

    cluster_id = 0

    for i in range(N):
        if labels[i] != UNVISITED:
            continue

        n_count = _region_query(
            data, i, eps_sq, spatial_h, spatial_w, spatial_weight, max_spatial_diff, neighbor_buf
        )

        if n_count < min_samples:
            labels[i] = NOISE
        else:
            labels[i] = cluster_id
            
            head = 0
            tail = 0
            for k in range(n_count):
                nb_idx = neighbor_buf[k]
                if nb_idx != i:
                    if labels[nb_idx] == UNVISITED or labels[nb_idx] == NOISE:
                        labels[nb_idx] = cluster_id
                        queue[tail] = nb_idx
                        tail += 1

            while head < tail:
                curr_p = queue[head]
                head += 1

                sub_count = _region_query(
                    data, curr_p, eps_sq, spatial_h, spatial_w, spatial_weight, max_spatial_diff, sub_neighbor_buf
                )

                if sub_count >= min_samples:
                    for k in range(sub_count):
                        sub_idx = sub_neighbor_buf[k]
                        if labels[sub_idx] == UNVISITED:
                            labels[sub_idx] = cluster_id
                            queue[tail] = sub_idx
                            tail += 1
                        elif labels[sub_idx] == NOISE:
                            labels[sub_idx] = cluster_id

            cluster_id += 1

    return labels

def visualize_dbscan_result(labels: np.ndarray) -> Tuple[np.ndarray, int, int]:
    '''
    Generate a colored visualization of the DBSCAN cluster label grid.

    Args:
        labels (np.ndarray): Assigned labels grid of shape (H, W).

    Returns:
        Tuple[np.ndarray, int, int]: Colored RGB image, number of clusters, and noise pixel count.
    '''
    import matplotlib.pyplot as plt
    H, W = labels.shape
    vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    clusters = [l for l in unique_labels if l >= 0]
    n_clusters = len(clusters)

    cmap = plt.get_cmap('tab20', max(n_clusters, 1))
    
    for i, cluster_id in enumerate(clusters):
        color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        vis_rgb[labels == cluster_id] = color

    vis_rgb[labels == -1] = [0, 0, 0]

    return vis_rgb, n_clusters, np.sum(labels == -1)


def compute_dbscan(data: Union[np.ndarray, torch.Tensor], eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    '''
    Perform standard DBSCAN clustering on generic feature matrix.

    Args:
        data (Union[np.ndarray, torch.Tensor]): Features matrix of shape (N, D).
        eps (float): Epsilon neighborhood search distance.
        min_samples (int): Minimum points required to form a core point cluster.

    Returns:
        np.ndarray: Labels array of shape (N,).
    '''
    if isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data)

    X = X.astype(np.float32)
    labels = _dbscan_numba(X, eps=eps, min_samples=min_samples, spatial_h=0, spatial_w=0, spatial_weight=0.0)
    return labels

def apply_dbscan(
    image: Union[np.ndarray, torch.Tensor],
    eps: float = 6.0,
    min_samples: int = 15,
    use_lab: bool = True,
    spatial_weight: float = 0.0,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply DBSCAN color/spatial clustering segmentation on an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        eps (float): Epsilon search distance.
        min_samples (int): Minimum points to form core clusters.
        use_lab (bool): Whether to perform clustering in LAB color space.
        spatial_weight (float): Multiplier weight for spatial coordinates.
        device (str): Device to use for PyTorch operations.

    Returns:
        np.ndarray: Labels grid array of shape (H, W).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    data_flat, H, W = preprocess_dbscan_pytorch(
        img_tensor, use_lab=use_lab, spatial_weight=spatial_weight, device=device
    )

    flat_labels = _dbscan_numba(
        data_flat, 
        eps=eps, 
        min_samples=min_samples, 
        spatial_h=H, 
        spatial_w=W, 
        spatial_weight=spatial_weight
    )

    return flat_labels.reshape(H, W)
