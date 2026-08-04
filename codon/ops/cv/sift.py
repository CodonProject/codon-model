from codon import *


def _gaussian_blur_2d(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    '''
    Apply separable 2D Gaussian blur on a PyTorch image tensor.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        torch.Tensor: Blurred image tensor.
    '''
    radius = int(np.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    x = torch.arange(kernel_size, device=img_tensor.device) - radius
    kernel_1d = torch.exp(-x**2 / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    k_x = kernel_1d.view(1, 1, 1, -1)
    k_y = kernel_1d.view(1, 1, -1, 1)

    pad = radius
    out = F.pad(img_tensor, (pad, pad, pad, pad), mode='reflect')
    out = F.conv2d(out, k_x)
    out = F.conv2d(out, k_y)
    return out

def build_pyramids_pytorch(
    img_tensor: torch.Tensor,
    n_octaves: int = 4,
    n_scales: int = 3,
    sigma: float = 1.6,
    device: str = 'cpu'
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    '''
    Build Gaussian and Difference-of-Gaussian (DoG) pyramids using PyTorch.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W) or (1, C, H, W).
        n_octaves (int): Number of octaves in scale space.
        n_scales (int): Number of scales per octave.
        sigma (float): Initial Gaussian blur sigma.
        device (str): Device to compute pyramids on.

    Returns:
        Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
            Gaussian pyramid and Difference-of-Gaussian pyramid.
    '''
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    img_tensor = img_tensor.to(device).float()
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    k = 2.0 ** (1.0 / n_scales)
    
    gaussian_pyramid = []
    dog_pyramid = []

    curr_img = img_tensor

    for octave in range(n_octaves):
        octave_gaussians = []
        sigmas = [sigma * (k ** i) for i in range(n_scales + 3)]
        
        for i in range(n_scales + 3):
            if i == 0 and octave == 0:
                blurred = _gaussian_blur_2d(curr_img, sigmas[0])
            elif i == 0:
                blurred = curr_img
            else:
                sigma_diff = np.sqrt(sigmas[i]**2 - sigmas[i-1]**2)
                blurred = _gaussian_blur_2d(octave_gaussians[-1], sigma_diff)
            
            octave_gaussians.append(blurred)

        octave_dogs = []
        for i in range(n_scales + 2):
            dog = octave_gaussians[i+1] - octave_gaussians[i]
            octave_dogs.append(dog.squeeze().cpu().numpy())
            
        gaussian_pyramid.append([g.squeeze().cpu().numpy() for g in octave_gaussians])
        dog_pyramid.append(octave_dogs)

        next_img = octave_gaussians[-3][:, :, ::2, ::2]
        curr_img = next_img

    return gaussian_pyramid, dog_pyramid

@numba.jit(nopython=True, fastmath=True)
def _solve_3x3(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, bool]:
    '''
    Solve linear system A x = b for a 3x3 matrix using Cramer's rule.

    Args:
        A (np.ndarray): 3x3 coefficient matrix.
        b (np.ndarray): 3D vector.

    Returns:
        Tuple[np.ndarray, bool]: Solution vector x and success flag.
    '''
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
    a20, a21, a22 = A[2, 0], A[2, 1], A[2, 2]
    b0, b1, b2    = b[0],    b[1],    b[2]

    detA = (a00 * (a11 * a22 - a12 * a21) - 
            a01 * (a10 * a22 - a12 * a20) + 
            a02 * (a10 * a21 - a11 * a20))

    if abs(detA) < 1e-10:
        return np.zeros(3, dtype=np.float32), False

    detX = (b0  * (a11 * a22 - a12 * a21) - 
            a01 * (b1  * a22 - a12 * b2)  + 
            a02 * (b1  * a21 - a11 * b2))

    detY = (a00 * (b1  * a22 - a12 * b2)  - 
            b0  * (a10 * a22 - a12 * a20) + 
            a02 * (a10 * b2  - b1  * a20))

    detZ = (a00 * (a11 * b2  - b1  * a21) - 
            a01 * (a10 * b2  - b1  * a20) + 
            b0  * (a10 * a21 - a11 * a20))

    x0 = detX / detA
    x1 = detY / detA
    x2 = detZ / detA

    res = np.array([x0, x1, x2], dtype=np.float32)
    return res, True

@numba.jit(nopython=True, fastmath=True)
def _refine_keypoint(
    dog_octave: List[np.ndarray],
    s: int,
    y: int,
    x: int,
    contrast_thresh: float,
    edge_thresh: float
) -> Tuple[bool, float, float, float, float]:
    '''
    Refine SIFT keypoint position and scale to sub-pixel accuracy.

    Args:
        dog_octave (List[np.ndarray]): DoG scale images in current octave.
        s (int): Scale index.
        y (int): Y coordinate.
        x (int): X coordinate.
        contrast_thresh (float): Contrast threshold for keypoint rejection.
        edge_thresh (float): Edge response ratio threshold.

    Returns:
        Tuple[bool, float, float, float, float]:
            Success flag, refined x, refined y, refined s, and refined contrast value.
    '''
    max_steps = 5
    img_h, img_w = dog_octave[0].shape

    offset = np.zeros(3, dtype=np.float32)
    dx, dy, ds = 0.0, 0.0, 0.0

    for step in range(max_steps):
        if s < 1 or s > len(dog_octave) - 2 or y < 5 or y >= img_h - 5 or x < 5 or x >= img_w - 5:
            return False, 0.0, 0.0, 0.0, 0.0

        dx = (dog_octave[s][y, x+1] - dog_octave[s][y, x-1]) * 0.5
        dy = (dog_octave[s][y+1, x] - dog_octave[s][y-1, x]) * 0.5
        ds = (dog_octave[s+1][y, x] - dog_octave[s-1][y, x]) * 0.5
        J = np.array([-dx, -dy, -ds], dtype=np.float32)

        dxx = dog_octave[s][y, x+1] + dog_octave[s][y, x-1] - 2.0 * dog_octave[s][y, x]
        dyy = dog_octave[s][y+1, x] + dog_octave[s][y-1, x] - 2.0 * dog_octave[s][y, x]
        dss = dog_octave[s+1][y, x] + dog_octave[s-1][y, x] - 2.0 * dog_octave[s][y, x]

        dxy = (dog_octave[s][y+1, x+1] - dog_octave[s][y+1, x-1] - dog_octave[s][y-1, x+1] + dog_octave[s][y-1, x-1]) * 0.25
        dxs = (dog_octave[s+1][y, x+1] - dog_octave[s+1][y, x-1] - dog_octave[s-1][y, x+1] + dog_octave[s-1][y, x-1]) * 0.25
        dys = (dog_octave[s+1][y+1, x] - dog_octave[s+1][y-1, x] - dog_octave[s-1][y+1, x] + dog_octave[s-1][y-1, x]) * 0.25

        H = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]
        ], dtype=np.float32)

        offset, ok = _solve_3x3(H, J)
        if not ok:
            return False, 0.0, 0.0, 0.0, 0.0

        if np.max(np.abs(offset)) < 0.5:
            break

        x += int(np.round(offset[0]))
        y += int(np.round(offset[1]))
        s += int(np.round(offset[2]))

    contrast = dog_octave[s][y, x] + 0.5 * (dx * offset[0] + dy * offset[1] + ds * offset[2])
    if abs(contrast) < contrast_thresh:
        return False, 0.0, 0.0, 0.0, 0.0

    trH = dxx + dyy
    detH = dxx * dyy - dxy * dxy
    if detH <= 0 or (trH * trH) / detH >= ((edge_thresh + 1.0) ** 2) / edge_thresh:
        return False, 0.0, 0.0, 0.0, 0.0

    return True, x + offset[0], y + offset[1], s + offset[2], contrast

@numba.jit(nopython=True, fastmath=True)
def _assign_orientation(
    img: np.ndarray,
    x: float,
    y: float,
    scale_sigma: float
) -> List[float]:
    '''
    Assign dominant orientation(s) to a keypoint.

    Args:
        img (np.ndarray): Gaussian image at keypoint scale.
        x (float): Sub-pixel X coordinate.
        y (float): Sub-pixel Y coordinate.
        scale_sigma (float): Keypoint scale sigma.

    Returns:
        List[float]: List of orientation angles in radians.
    '''
    H, W = img.shape
    radius = int(np.ceil(3.0 * 1.5 * scale_sigma))
    hist = np.zeros(36, dtype=np.float32)
    sig_sq = 2.0 * (1.5 * scale_sigma) ** 2

    ix, iy = int(np.round(x)), int(np.round(y))

    for dy in range(-radius, radius + 1):
        py = iy + dy
        if py <= 0 or py >= H - 1:
            continue
        for dx in range(-radius, radius + 1):
            px = ix + dx
            if px <= 0 or px >= W - 1:
                continue

            dist_sq = dx * dx + dy * dy
            weight = np.exp(-dist_sq / sig_sq)

            gx = img[py, px + 1] - img[py, px - 1]
            gy = img[py + 1, px] - img[py - 1, px]
            mag = np.sqrt(gx * gx + gy * gy)
            angle = np.arctan2(gy, gx) % (2.0 * np.pi)

            bin_idx = int(np.floor(36.0 * angle / (2.0 * np.pi))) % 36
            hist[bin_idx] += weight * mag

    max_val = np.max(hist)
    orientations = []

    for i in range(36):
        prev_v = hist[(i - 1) % 36]
        curr_v = hist[i]
        next_v = hist[(i + 1) % 36]

        if curr_v > prev_v and curr_v > next_v and curr_v >= 0.8 * max_val:
            interp_bin = i + 0.5 * (prev_v - next_v) / (prev_v - 2.0 * curr_v + next_v)
            angle = (interp_bin % 36) * (2.0 * np.pi / 36.0)
            orientations.append(angle)

    return orientations

@numba.jit(nopython=True, fastmath=True)
def _compute_sift_descriptor(
    img: np.ndarray,
    x: float,
    y: float,
    main_angle: float,
    scale_sigma: float,
    d: int = 4,
    n_bins: int = 8
) -> np.ndarray:
    '''
    Compute 128-dimensional SIFT descriptor vector for a keypoint.

    Args:
        img (np.ndarray): Gaussian image patch.
        x (float): Keypoint X coordinate.
        y (float): Keypoint Y coordinate.
        main_angle (float): Keypoint orientation angle in radians.
        scale_sigma (float): Keypoint scale sigma.
        d (int): Width of keypoint descriptor grid (default 4 for 4x4).
        n_bins (int): Number of orientation histogram bins per sub-region (default 8).

    Returns:
        np.ndarray: Normalized descriptor vector of size d*d*n_bins.
    '''
    H, W = img.shape
    cos_a = np.cos(-main_angle)
    sin_a = np.sin(-main_angle)
    
    hist_width = 3.0 * scale_sigma
    radius = int(np.ceil(np.sqrt(2.0) * hist_width * (d + 1) * 0.5))

    descriptor = np.zeros((d, d, n_bins), dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rx = (dx * cos_a - dy * sin_a) / hist_width
            ry = (dx * sin_a + dy * cos_a) / hist_width

            r_bin = ry + d / 2.0 - 0.5
            c_bin = rx + d / 2.0 - 0.5

            if -1.0 < r_bin < d and -1.0 < c_bin < d:
                px = int(np.round(x)) + dx
                py = int(np.round(y)) + dy

                if 0 < px < W - 1 and 0 < py < H - 1:
                    gx = img[py, px + 1] - img[py, px - 1]
                    gy = img[py + 1, px] - img[py - 1, px]
                    mag = np.sqrt(gx * gx + gy * gy)
                    angle = (np.arctan2(gy, gx) - main_angle) % (2.0 * np.pi)

                    o_bin = angle * (n_bins / (2.0 * np.pi))

                    w = np.exp(-(rx * rx + ry * ry) / (0.5 * d * d)) * mag

                    r0 = int(np.floor(r_bin))
                    c0 = int(np.floor(c_bin))
                    o0 = int(np.floor(o_bin))

                    dr = r_bin - r0
                    dc = c_bin - c0
                    do = o_bin - o0

                    for dr_i in range(2):
                        r_idx = r0 + dr_i
                        if 0 <= r_idx < d:
                            wr = dr if dr_i == 1 else (1.0 - dr)
                            for dc_i in range(2):
                                c_idx = c0 + dc_i
                                if 0 <= c_idx < d:
                                    wc = dc if dc_i == 1 else (1.0 - dc)
                                    for do_i in range(2):
                                        o_idx = (o0 + do_i) % n_bins
                                        wo = do if do_i == 1 else (1.0 - do)
                                        
                                        descriptor[r_idx, c_idx, o_idx] += w * wr * wc * wo

    vec = descriptor.ravel()

    norm = np.sqrt(np.sum(vec**2)) + 1e-7
    vec /= norm

    vec = np.minimum(vec, 0.2)

    norm = np.sqrt(np.sum(vec**2)) + 1e-7
    vec /= norm

    return vec

def apply_sift(
    image: Union[np.ndarray, torch.Tensor],
    n_octaves: int = 4,
    n_scales: int = 3,
    sigma: float = 1.6,
    contrast_thresh: float = 0.04,
    edge_thresh: float = 10.0,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Detect SIFT keypoints and compute 128-dimensional descriptors.

    Args:
        image (Union[np.ndarray, torch.Tensor]): Input image array or tensor.
        n_octaves (int): Number of octaves.
        n_scales (int): Number of scales per octave.
        sigma (float): Base Gaussian smoothing parameter.
        contrast_thresh (float): Minimum contrast threshold for keypoints.
        edge_thresh (float): Edge response ratio threshold.
        device (str): Computation device for PyTorch.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Keypoints array of shape (N, 4) containing [x, y, sigma, orientation].
            - Descriptors array of shape (N, 128).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    gaussian_pyramid, dog_pyramid = build_pyramids_pytorch(
        img_tensor, n_octaves=n_octaves, n_scales=n_scales, sigma=sigma, device=device
    )

    keypoints = []
    descriptors = []

    k = 2.0 ** (1.0 / n_scales)

    for octave_idx in range(n_octaves):
        dog_octave = dog_pyramid[octave_idx]
        gauss_octave = gaussian_pyramid[octave_idx]
        
        scale_factor = 2.0 ** octave_idx

        for s in range(1, n_scales + 1):
            bot, mid, top = dog_octave[s-1], dog_octave[s], dog_octave[s+1]
            img_h, img_w = mid.shape

            for y in range(5, img_h - 5):
                for x in range(5, img_w - 5):
                    val = mid[y, x]

                    if abs(val) < 0.8 * contrast_thresh:
                        continue

                    patch_bot = bot[y-1:y+2, x-1:x+2]
                    patch_mid = mid[y-1:y+2, x-1:x+2]
                    patch_top = top[y-1:y+2, x-1:x+2]

                    is_max = (val > 0) and (val >= np.max(patch_bot)) and (val >= np.max(patch_top)) and (val >= np.max(patch_mid))
                    is_min = (val < 0) and (val <= np.min(patch_bot)) and (val <= np.min(patch_top)) and (val <= np.min(patch_mid))

                    if is_max or is_min:
                        success, fx, fy, fs, contrast = _refine_keypoint(
                            dog_octave, s, y, x, contrast_thresh, edge_thresh
                        )

                        if success:
                            curr_sigma = sigma * (k ** fs)
                            
                            orientations = _assign_orientation(gauss_octave[int(np.round(fs))], fx, fy, curr_sigma)

                            for angle in orientations:
                                desc = _compute_sift_descriptor(
                                    gauss_octave[int(np.round(fs))], fx, fy, angle, curr_sigma
                                )
                                
                                real_x = fx * scale_factor
                                real_y = fy * scale_factor
                                real_sigma = curr_sigma * scale_factor

                                keypoints.append([real_x, real_y, real_sigma, angle])
                                descriptors.append(desc)

    if len(keypoints) == 0:
        return np.empty((0, 4)), np.empty((0, 128))

    return np.array(keypoints, dtype=np.float32), np.array(descriptors, dtype=np.float32)
