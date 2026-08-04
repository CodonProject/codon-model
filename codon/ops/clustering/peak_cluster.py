import torch
import numpy as np
from numba import jit

def _rgb_to_lab_pytorch(img_tensor):
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

def preprocess_quickshift_pytorch(img_tensor, ratio=1.0, device='cpu'):
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
    elif img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)

    img_tensor = img_tensor.to(device).float()
    
    lab_tensor = _rgb_to_lab_pytorch(img_tensor)
    H, W = lab_tensor.shape[1], lab_tensor.shape[2]

    L = lab_tensor[0]
    a = lab_tensor[1]
    b = lab_tensor[2]

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    features = torch.stack([
        L, a, b,
        y_grid * ratio,
        x_grid * ratio
    ], dim=-1)

    return features.cpu().numpy()

@jit(nopython=True, fastmath=True)
def _compute_density(features, kernel_size):
    H, W, _ = features.shape
    densities = np.zeros((H, W), dtype=np.float32)
    
    radius = max(1, int(np.ceil(3.0 * kernel_size)))
    inv_sigma2 = 1.0 / (2.0 * kernel_size * kernel_size)

    for y in range(H):
        y_min = max(0, y - radius)
        y_max = min(H, y + radius + 1)
        
        for x in range(W):
            x_min = max(0, x - radius)
            x_max = min(W, x + radius + 1)

            f0, f1, f2, f3, f4 = features[y, x, 0], features[y, x, 1], features[y, x, 2], features[y, x, 3], features[y, x, 4]
            dens = 0.0

            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    d0 = f0 - features[ny, nx, 0]
                    d1 = f1 - features[ny, nx, 1]
                    d2 = f2 - features[ny, nx, 2]
                    d3 = f3 - features[ny, nx, 3]
                    d4 = f4 - features[ny, nx, 4]

                    dist_sq = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4
                    dens += np.exp(-dist_sq * inv_sigma2)

            densities[y, x] = dens

    return densities

@jit(nopython=True, fastmath=True)
def _quickshift_find_parents(features, densities, kernel_size, max_dist):
    H, W, _ = features.shape
    
    search_radius = max(int(np.ceil(max_dist)), int(np.ceil(3.0 * kernel_size)))
    max_dist_sq = max_dist * max_dist

    parents = np.zeros((H, W), dtype=np.int32)

    for y in range(H):
        y_min = max(0, y - search_radius)
        y_max = min(H, y + search_radius + 1)

        for x in range(W):
            x_min = max(0, x - search_radius)
            x_max = min(W, x + search_radius + 1)

            curr_dens = densities[y, x]
            curr_flat_idx = y * W + x

            f0, f1, f2, f3, f4 = features[y, x, 0], features[y, x, 1], features[y, x, 2], features[y, x, 3], features[y, x, 4]

            min_dist_sq = 1e9
            best_parent = curr_flat_idx

            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    neigh_dens = densities[ny, nx]
                    neigh_flat_idx = ny * W + nx

                    is_higher_density = (neigh_dens > curr_dens) or (abs(neigh_dens - curr_dens) < 1e-6 and neigh_flat_idx < curr_flat_idx)

                    if is_higher_density:
                        d0 = f0 - features[ny, nx, 0]
                        d1 = f1 - features[ny, nx, 1]
                        d2 = f2 - features[ny, nx, 2]
                        d3 = f3 - features[ny, nx, 3]
                        d4 = f4 - features[ny, nx, 4]

                        dist_sq = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4

                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_parent = neigh_flat_idx

            if min_dist_sq > max_dist_sq:
                best_parent = curr_flat_idx

            parents[y, x] = best_parent

    return parents

@jit(nopython=True, fastmath=True)
def _flat_tree_segmentation(parents):
    H, W = parents.shape
    N = H * W
    flat_parents = parents.ravel()

    for i in range(N):
        root = i
        while flat_parents[root] != root:
            root = flat_parents[root]
        
        curr = i
        while curr != root:
            next_p = flat_parents[curr]
            flat_parents[curr] = root
            curr = next_p

    label_map = np.full(N, -1, dtype=np.int32)
    labels = np.zeros((H, W), dtype=np.int32)
    
    current_label = 0
    for i in range(N):
        root = flat_parents[i]
        if label_map[root] == -1:
            label_map[root] = current_label
            current_label += 1
        
        y = i // W
        x = i % W
        labels[y, x] = label_map[root]

    return labels

@jit(nopython=True, fastmath=True)
def _compute_density_nd(data, kernel_size):
    N, D = data.shape
    densities = np.zeros(N, dtype=np.float32)
    inv_sigma2 = 1.0 / (2.0 * kernel_size * kernel_size)

    for i in range(N):
        dens = 0.0
        for j in range(N):
            dist_sq = 0.0
            for d in range(D):
                diff = data[i, d] - data[j, d]
                dist_sq += diff * diff
            dens += np.exp(-dist_sq * inv_sigma2)
        densities[i] = dens
    return densities

@jit(nopython=True, fastmath=True)
def _quickshift_find_parents_nd(data, densities, max_dist):
    N, D = data.shape
    max_dist_sq = max_dist * max_dist
    parents = np.zeros(N, dtype=np.int32)

    for i in range(N):
        curr_dens = densities[i]
        min_dist_sq = 1e9
        best_parent = i

        for j in range(N):
            neigh_dens = densities[j]
            is_higher_density = (neigh_dens > curr_dens) or (abs(neigh_dens - curr_dens) < 1e-6 and j < i)

            if is_higher_density:
                dist_sq = 0.0
                for d in range(D):
                    diff = data[i, d] - data[j, d]
                    dist_sq += diff * diff

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_parent = j

        if min_dist_sq > max_dist_sq:
            best_parent = i

        parents[i] = best_parent
    return parents

@jit(nopython=True, fastmath=True)
def _flat_tree_segmentation_nd(parents):
    N = len(parents)
    flat_parents = parents.copy()

    for i in range(N):
        root = i
        while flat_parents[root] != root:
            root = flat_parents[root]
        
        curr = i
        while curr != root:
            next_p = flat_parents[curr]
            flat_parents[curr] = root
            curr = next_p

    label_map = np.full(N, -1, dtype=np.int32)
    labels = np.zeros(N, dtype=np.int32)
    
    current_label = 0
    for i in range(N):
        root = flat_parents[i]
        if label_map[root] == -1:
            label_map[root] = current_label
            current_label += 1
        labels[i] = label_map[root]

    return labels

def compute_peak_clustering(data, kernel_size=2.0, max_dist=10.0):
    if isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data)

    X = X.astype(np.float32)
    densities = _compute_density_nd(X, kernel_size=kernel_size)
    parents = _quickshift_find_parents_nd(X, densities, max_dist=max_dist)
    labels = _flat_tree_segmentation_nd(parents)
    return labels

def apply_peak_clustering(image, ratio=0.5, kernel_size=2.0, max_dist=10.0, device='cpu'):
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image
        if img_tensor.ndim == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

    features = preprocess_quickshift_pytorch(img_tensor, ratio=ratio, device=device)
    densities = _compute_density(features, kernel_size=kernel_size)
    parents = _quickshift_find_parents(features, densities, kernel_size=kernel_size, max_dist=max_dist)
    labels = _flat_tree_segmentation(parents)

    return labels