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

def preprocess_kmeans_pytorch(img_tensor, use_lab=True, spatial_weight=0.0, device='cpu'):
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

@jit(nopython=True, fastmath=True)
def _kmeans_pp_init(data, K, seed=42):
    np.random.seed(seed)
    N, D = data.shape
    centers = np.zeros((K, D), dtype=np.float32)

    first_idx = np.random.randint(0, N)
    for d in range(D):
        centers[0, d] = data[first_idx, d]

    dist_sq = np.full(N, 1e9, dtype=np.float32)

    for k in range(1, K):
        prev_center = centers[k - 1]
        
        for i in range(N):
            d2 = 0.0
            for d in range(D):
                diff = data[i, d] - prev_center[d]
                d2 += diff * diff
            if d2 < dist_sq[i]:
                dist_sq[i] = d2

        total_dist = np.sum(dist_sq)

        if total_dist <= 1e-8:
            rand_idx = np.random.randint(0, N)
            for d in range(D):
                centers[k, d] = data[rand_idx, d]
        else:
            rand_val = np.random.uniform(0.0, total_dist)
            cum_sum = 0.0
            selected_idx = 0
            for i in range(N):
                cum_sum += dist_sq[i]
                if cum_sum >= rand_val:
                    selected_idx = i
                    break
            
            for d in range(D):
                centers[k, d] = data[selected_idx, d]

    return centers

@jit(nopython=True, fastmath=True)
def _kmeans_random_init(data, K, seed=42):
    np.random.seed(seed)
    N, D = data.shape
    indices = np.random.choice(N, K, replace=False)
    centers = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        idx = indices[k]
        for d in range(D):
            centers[k, d] = data[idx, d]
    return centers

@jit(nopython=True, fastmath=True)
def _kmeans_lloyd(data, centers, max_iter=100, tol=1e-4):
    N, D = data.shape
    K = centers.shape[0]

    labels = np.zeros(N, dtype=np.int32)
    counts = np.zeros(K, dtype=np.float32)
    new_centers = np.zeros((K, D), dtype=np.float32)

    for it in range(max_iter):
        for i in range(N):
            min_d2 = 1e9
            best_k = 0
            for k in range(K):
                d2 = 0.0
                for d in range(D):
                    diff = data[i, d] - centers[k, d]
                    d2 += diff * diff
                if d2 < min_d2:
                    min_d2 = d2
                    best_k = k
            labels[i] = best_k

        new_centers.fill(0.0)
        counts.fill(0.0)

        for i in range(N):
            k = labels[i]
            counts[k] += 1.0
            for d in range(D):
                new_centers[k, d] += data[i, d]

        shift = 0.0
        for k in range(K):
            if counts[k] > 0.0:
                for d in range(D):
                    new_centers[k, d] /= counts[k]
            else:
                for d in range(D):
                    new_centers[k, d] = centers[k, d]

            d2 = 0.0
            for d in range(D):
                diff = new_centers[k, d] - centers[k, d]
                d2 += diff * diff
            shift += np.sqrt(d2)

        for k in range(K):
            for d in range(D):
                centers[k, d] = new_centers[k, d]

        if shift < tol:
            break

    return labels, centers

# ==========================================
# 纯数据接口: compute_kmeans
# ==========================================
def compute_kmeans(data, n_clusters=5, init='kmeans++', max_iter=100, tol=1e-4, seed=42):
    """
    对纯特征矩阵 X (N, D) 执行 K-Means / K-Means++ 聚类
    :param data: NumPy 数组或 PyTorch Tensor，形状为 (N, D)
    :param n_clusters: 聚类簇数 K
    :param init: 初始化策略 'kmeans++' 或 'random'
    :param max_iter: 最大迭代次数
    :param tol: 中心位移收敛阈值
    :param seed: 随机种子
    :return: labels (N,), centers (K, D)
    """
    if isinstance(data, torch.Tensor):
        X = data.detach().cpu().numpy()
    else:
        X = np.asarray(data)

    X = X.astype(np.float32)

    if init.lower() == 'kmeans++':
        initial_centers = _kmeans_pp_init(X, K=n_clusters, seed=seed)
    else:
        initial_centers = _kmeans_random_init(X, K=n_clusters, seed=seed)

    labels, centers = _kmeans_lloyd(X, initial_centers, max_iter=max_iter, tol=tol)
    return labels, centers

def apply_kmeans(
    image, 
    n_clusters=5, 
    init='kmeans++', 
    max_iter=100, 
    tol=1e-4, 
    use_lab=True, 
    spatial_weight=0.0, 
    seed=42, 
    device='cpu'
):
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    data_flat, H, W = preprocess_kmeans_pytorch(
        img_tensor, use_lab=use_lab, spatial_weight=spatial_weight, device=device
    )

    flat_labels, final_centers = compute_kmeans(
        data_flat, n_clusters=n_clusters, init=init, max_iter=max_iter, tol=tol, seed=seed
    )

    labels = flat_labels.reshape(H, W)

    return labels, final_centers