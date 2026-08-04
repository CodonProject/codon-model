from codon import *


def compute_template_matching_map(
    image_tensor: torch.Tensor,
    template_tensor: torch.Tensor,
    method: str = 'CCOEFF_NORMED',
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Compute template matching response map using PyTorch.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W), or (1, C, H, W).
        template_tensor (torch.Tensor): Template image tensor of shape (h, w), (C, h, w), or (1, C, h, w).
        method (str): Matching method. Supported methods: 'CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        np.ndarray: 2D response map of shape (H - h + 1, W - w + 1).

    Raises:
        ValueError: If the number of channels of the image and template do not match, or if the method is unsupported.
    '''
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if template_tensor.dim() == 2:
        template_tensor = template_tensor.unsqueeze(0).unsqueeze(0)
    elif template_tensor.dim() == 3:
        template_tensor = template_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device).float()
    template_tensor = template_tensor.to(device).float()

    N, C, H, W = image_tensor.shape
    _, C_t, h, w = template_tensor.shape
    assert C == C_t, f"图像通道数 ({C}) 与模板通道数 ({C_t}) 不匹配！"

    ones_kernel = torch.ones((1, C, h, w), device=device, dtype=torch.float32)

    method = method.upper()

    if method == 'CCOEFF_NORMED':
        t_mean = template_tensor.mean(dim=(2, 3), keepdim=True)
        t_zero = template_tensor - t_mean
        t_sum_sq = (t_zero ** 2).sum()

        i_sum = F.conv2d(image_tensor, ones_kernel)
        i_bar = i_sum / (C * h * w)

        i_sq_sum = F.conv2d(image_tensor ** 2, ones_kernel)
        i_zero_sum_sq = torch.clamp(i_sq_sum - (C * h * w) * (i_bar ** 2), min=1e-10)

        numerator = F.conv2d(image_tensor, t_zero)

        denominator = torch.sqrt(i_zero_sum_sq * t_sum_sq)
        response_map = numerator / torch.clamp(denominator, min=1e-8)

    elif method == 'CCORR_NORMED':
        t_sum_sq = (template_tensor ** 2).sum()
        i_sq_sum = F.conv2d(image_tensor ** 2, ones_kernel)

        numerator = F.conv2d(image_tensor, template_tensor)
        denominator = torch.sqrt(torch.clamp(i_sq_sum * t_sum_sq, min=1e-10))
        response_map = numerator / torch.clamp(denominator, min=1e-8)

    elif method == 'SQDIFF_NORMED':
        t_sum_sq = (template_tensor ** 2).sum()
        i_sq_sum = F.conv2d(image_tensor ** 2, ones_kernel)
        cross_term = F.conv2d(image_tensor, template_tensor)

        numerator = torch.clamp(i_sq_sum + t_sum_sq - 2.0 * cross_term, min=0.0)
        denominator = torch.sqrt(torch.clamp(i_sq_sum * t_sum_sq, min=1e-10))
        response_map = numerator / torch.clamp(denominator, min=1e-8)

    else:
        raise ValueError(f"不支持的匹配方法: {method}")

    return response_map.squeeze().cpu().numpy()


@numba.jit(nopython=True, fastmath=True)
def _nms_2d_peaks(
    response_map: np.ndarray,
    h: int,
    w: int,
    threshold: float = 0.8,
    max_matches: int = 100,
    is_sqdiff: bool = False
) -> np.ndarray:
    '''
    Perform 2D Non-Maximum Suppression (NMS) to detect peaks in template matching response map.

    Args:
        response_map (np.ndarray): Response map array of shape (H_map, W_map).
        h (int): Height of the template.
        w (int): Width of the template.
        threshold (float): Score threshold for peaks.
        max_matches (int): Maximum number of matched locations to return.
        is_sqdiff (bool): True if SQDIFF_NORMED is used (lower is better), False otherwise.

    Returns:
        np.ndarray: Filtered bounding boxes array of shape (M, 5), where each row is [x1, y1, x2, y2, score].
    '''
    H_map, W_map = response_map.shape
    candidates = []

    for y in range(H_map):
        for x in range(W_map):
            score = response_map[y, x]
            if is_sqdiff:
                if score <= threshold:
                    candidates.append((score, x, y))
            else:
                if score >= threshold:
                    candidates.append((score, x, y))

    if len(candidates) == 0:
        return np.empty((0, 5), dtype=np.float32)

    n_cand = len(candidates)
    scores = np.zeros(n_cand, dtype=np.float32)
    xs = np.zeros(n_cand, dtype=np.int32)
    ys = np.zeros(n_cand, dtype=np.int32)

    for i in range(n_cand):
        scores[i] = candidates[i][0]
        xs[i] = candidates[i][1]
        ys[i] = candidates[i][2]

    if is_sqdiff:
        sort_indices = np.argsort(scores)
    else:
        sort_indices = np.argsort(-scores)

    suppressed = np.zeros(n_cand, dtype=numba.boolean)
    boxes = []

    half_w = w / 2.0
    half_h = h / 2.0

    for i in range(n_cand):
        idx = sort_indices[i]
        if suppressed[idx]:
            continue

        cx = xs[idx]
        cy = ys[idx]
        sc = scores[idx]

        x1 = float(cx)
        y1 = float(cy)
        x2 = float(cx + w)
        y2 = float(cy + h)

        boxes.append([x1, y1, x2, y2, sc])
        if len(boxes) >= max_matches:
            break

        for j in range(i + 1, n_cand):
            idx_j = sort_indices[j]
            if suppressed[idx_j]:
                continue

            cx_j = xs[idx_j]
            cy_j = ys[idx_j]

            if abs(cx - cx_j) < half_w and abs(cy - cy_j) < half_h:
                suppressed[idx_j] = True

    res = np.zeros((len(boxes), 5), dtype=np.float32)
    for i in range(len(boxes)):
        for j in range(5):
            res[i, j] = boxes[i][j]

    return res


def apply_template_matching(
    image: Union[np.ndarray, torch.Tensor],
    template: Union[np.ndarray, torch.Tensor],
    method: str = 'CCOEFF_NORMED',
    threshold: float = 0.8,
    max_matches: int = 100,
    return_map: bool = False,
    device: str = 'cpu'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    '''
    Apply standard template matching to locate occurrences of the template image in the scene image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The scene image.
        template (Union[np.ndarray, torch.Tensor]): The template image.
        method (str): Matching method. Supported methods: 'CCOEFF_NORMED', 'CCORR_NORMED', 'SQDIFF_NORMED'.
        threshold (float): Matching score threshold.
        max_matches (int): Maximum number of matched locations to return.
        return_map (bool): Whether to return the raw response map.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - If return_map is False: Bounding boxes array of shape (M, 5), where each row is [x1, y1, x2, y2, score].
            - If return_map is True: A tuple (boxes, response_map).
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    if isinstance(template, np.ndarray):
        tpl_tensor = torch.from_numpy(template)
    else:
        tpl_tensor = template

    if img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
        img_tensor = img_tensor.permute(2, 0, 1)
    if tpl_tensor.ndim == 3 and tpl_tensor.shape[2] in [1, 3]:
        tpl_tensor = tpl_tensor.permute(2, 0, 1)

    h = tpl_tensor.shape[-2]
    w = tpl_tensor.shape[-1]

    response_map = compute_template_matching_map(
        img_tensor, tpl_tensor, method=method, device=device
    )

    is_sqdiff = (method.upper() == 'SQDIFF_NORMED')
    boxes = _nms_2d_peaks(
        response_map, h=h, w=w, threshold=threshold, max_matches=max_matches, is_sqdiff=is_sqdiff
    )

    if return_map:
        return boxes, response_map
    return boxes
