import torch
import torch.nn.functional as F
import numpy as np
import numba
from numba import jit

def _gaussian_kernel(kernel_size=5, sigma=1.4):
    x = torch.arange(kernel_size) - kernel_size // 2
    grid = x.repeat(kernel_size, 1)
    kernel = torch.exp(-(grid**2 + grid.T**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

def preprocess_canny_pytorch(img_tensor, kernel_size=5, sigma=1.4, device='cpu'):
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    img_tensor = img_tensor.to(device).float()
    
    pad = kernel_size // 2
    kernel = _gaussian_kernel(kernel_size=kernel_size, sigma=sigma).to(device)
    img_blur = F.conv2d(img_tensor, kernel, padding=pad)
    
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    
    gx = F.conv2d(img_blur, sobel_x, padding=1).squeeze()
    gy = F.conv2d(img_blur, sobel_y, padding=1).squeeze()
    
    magnitude = torch.sqrt(gx**2 + gy**2)
    angles = torch.atan2(gy, gx)
    
    return magnitude.cpu().numpy(), angles.cpu().numpy()

@jit(nopython=True, fastmath=True)
def _non_max_suppression(mag, angle):
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

@jit(nopython=True, fastmath=True)
def _hysteresis_thresholding(nms_img, low_thresh, high_thresh):
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

def apply_canny(image, low_thresh=50.0, high_thresh=150.0, kernel_size=5, sigma=1.4, device='cpu'):
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    mag, angles = preprocess_canny_pytorch(img_tensor, kernel_size=kernel_size, sigma=sigma, device=device)
    nms_img = _non_max_suppression(mag, angles)
    edge_map = _hysteresis_thresholding(nms_img, low_thresh=low_thresh, high_thresh=high_thresh)

    return edge_map