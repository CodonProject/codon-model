import torch
import numpy as np
import numba
from numba import jit

def preprocess_hough_pytorch(img_tensor, edge_threshold=128.0, device='cpu'):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze()

    img_tensor = img_tensor.to(device).float()
    
    edge_indices = torch.nonzero(img_tensor >= edge_threshold)
    
    return edge_indices.cpu().numpy()

@jit(nopython=True, fastmath=True)
def _hough_accumulate(edge_coords, num_rhos, num_thetas, sin_t, cos_t, max_dist, rho_res):
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

@jit(nopython=True, fastmath=True)
def _find_hough_peaks(accumulator, threshold, nhood_r=5, nhood_t=5):
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

def hough_lines_to_endpoints(lines, img_shape):
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

def apply_hough(image, rho_res=1.0, theta_res=np.pi/180.0, threshold=50, edge_threshold=128.0, return_endpoints=False, device='cpu'):
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