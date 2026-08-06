from codon import *
from codon.ops import prepare_input_tensor, nms_2d_peaks


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
    image_tensor = prepare_input_tensor(image_tensor, device=device)
    template_tensor = prepare_input_tensor(template_tensor, device=device)

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
        raise ValueError(f"不支持的匹配 method: {method}")

    return response_map.squeeze().cpu().numpy()


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
    img_tensor = prepare_input_tensor(image, device=device)
    tpl_tensor = prepare_input_tensor(template, device=device)

    h, w = tpl_tensor.shape[-2:]

    response_map = compute_template_matching_map(
        img_tensor, tpl_tensor, method=method, device=device
    )

    is_sqdiff = (method.upper() == 'SQDIFF_NORMED')
    boxes = nms_2d_peaks(
        response_map, h=h, w=w, threshold=threshold, max_matches=max_matches, is_sqdiff=is_sqdiff
    )

    if return_map:
        return boxes, response_map
    return boxes
