from codon import *
from codon.ops import compute_image_gradients


def preprocess_shape_matching_pytorch(
    img_tensor: torch.Tensor,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Preprocess image tensor for shape matching.

    Convert input image tensor to grayscale, apply Gaussian blur, and compute normalized gradients.

    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (H, W), (C, H, W), or (1, C, H, W).
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - mag (np.ndarray): Gradient magnitude array of shape (H, W).
            - gx_norm (np.ndarray): Normalized horizontal gradient array of shape (H, W).
            - gy_norm (np.ndarray): Normalized vertical gradient array of shape (H, W).
    '''
    mag_tensor, gx_tensor, gy_tensor, _ = compute_image_gradients(
        img_tensor, blur_sigma=0.8, kernel_size=5, device=device
    )

    mag_clamp = torch.clamp(mag_tensor, min=1e-5)
    gx_norm = gx_tensor / mag_clamp
    gy_norm = gy_tensor / mag_clamp

    return (mag_tensor.cpu().numpy(),
            gx_norm.cpu().numpy(),
            gy_norm.cpu().numpy())


@numba.jit(nopython=True, fastmath=True)
def _trace_contour_moore(binary_mask: np.ndarray) -> List[np.ndarray]:
    '''
    Trace ordered boundary contour points using Moore-Neighbor Tracing algorithm.

    Args:
        binary_mask (np.ndarray): Boolean 2D binary mask array of shape (H, W).

    Returns:
        List[np.ndarray]: List of ordered contour point arrays, each of shape (N, 2) [x, y].
    '''
    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=numba.boolean)
    contours = []

    # 8-neighbor directions (clockwise)
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if binary_mask[y, x] and not visited[y, x]:
                # Check if it is a boundary pixel
                is_boundary = False
                for k in range(8):
                    if not binary_mask[y + dy[k], x + dx[k]]:
                        is_boundary = True
                        break

                if not is_boundary:
                    continue

                # Start Moore-Neighbor Tracing
                pts = []
                cx, cy = x, y
                entry_dir = 0

                curr_x, curr_y = cx, cy
                first_x, first_y = cx, cy
                second_x, second_y = -1, -1

                loop_counter = 0

                while loop_counter < H * W:
                    loop_counter += 1
                    pts.append((float(curr_x), float(curr_y)))
                    visited[curr_y, curr_x] = True

                    search_dir = (entry_dir + 5) % 8
                    found_next = False

                    for i in range(8):
                        d = (search_dir + i) % 8
                        nx = curr_x + dx[d]
                        ny = curr_y + dy[d]

                        if 0 <= nx < W and 0 <= ny < H:
                            if binary_mask[ny, nx]:
                                if len(pts) == 2:
                                    second_x, second_y = curr_x, curr_y

                                entry_dir = d
                                curr_x, curr_y = nx, ny
                                found_next = True
                                break

                    if not found_next:
                        break

                    if curr_x == first_x and curr_y == first_y and (second_x == -1 or len(pts) > 3):
                        break

                if len(pts) > 5:
                    pt_arr = np.zeros((len(pts), 2), dtype=np.float32)
                    for idx in range(len(pts)):
                        pt_arr[idx, 0] = pts[idx][0]
                        pt_arr[idx, 1] = pts[idx][1]
                    contours.append(pt_arr)

    return contours


def extract_template_ordered_contours(
    template_img: torch.Tensor,
    mag_thresh: float = 40.0,
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], int, int]:
    '''
    Extract ordered boundary contours with directional unit gradients from the template image.

    Args:
        template_img (torch.Tensor): Input template image tensor.
        mag_thresh (float): Threshold of gradient magnitude to build binary mask.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        Tuple[List[np.ndarray], int, int]: A tuple containing:
            - contours_features (List[np.ndarray]): List of feature arrays of shape (N, 4) where each row is [dx, dy, u, v].
            - h (int): Height of the template image.
            - w (int): Width of the template image.
    '''
    mag, gx_norm, gy_norm = preprocess_shape_matching_pytorch(template_img, device=device)
    h, w = mag.shape
    cx_center, cy_center = w / 2.0, h / 2.0

    binary_mask = mag >= mag_thresh
    raw_contours = _trace_contour_moore(binary_mask)

    contour_features = []
    for pt_arr in raw_contours:
        feat_arr = np.zeros((len(pt_arr), 4), dtype=np.float32)
        for i in range(len(pt_arr)):
            px = int(np.round(pt_arr[i, 0]))
            py = int(np.round(pt_arr[i, 1]))
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            feat_arr[i, 0] = pt_arr[i, 0] - cx_center
            feat_arr[i, 1] = pt_arr[i, 1] - cy_center
            feat_arr[i, 2] = gx_norm[py, px]
            feat_arr[i, 3] = gy_norm[py, px]

        contour_features.append(feat_arr)

    return contour_features, h, w


def extract_template_shape_features(
    template_img: torch.Tensor,
    mag_thresh: float = 50.0,
    max_points: int = 250,
    device: str = 'cpu'
) -> Tuple[np.ndarray, int, int]:
    '''
    Extract shape matching features from the template image.

    Args:
        template_img (torch.Tensor): Input template image tensor.
        mag_thresh (float): Threshold of gradient magnitude to filter edges.
        max_points (int): Maximum number of feature points to select.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, int, int]: A tuple containing:
            - features (np.ndarray): Array of shape (N, 4) where each row is [dx, dy, u, v].
            - h (int): Height of the template image.
            - w (int): Width of the template image.
    '''
    mag, gx_norm, gy_norm = preprocess_shape_matching_pytorch(template_img, device=device)

    edge_y, edge_x = np.where(mag >= mag_thresh)
    if len(edge_x) == 0:
        raise ValueError("模板中未提取到有效边缘，请降低 mag_thresh！")

    if len(edge_x) > max_points:
        step = len(edge_x) // max_points
        edge_x = edge_x[::step][:max_points]
        edge_y = edge_y[::step][:max_points]

    h, w = mag.shape
    cx, cy = w / 2.0, h / 2.0

    features = []
    for x, y in zip(edge_x, edge_y):
        dx = float(x - cx)
        dy = float(y - cy)
        u = float(gx_norm[y, x])
        v = float(gy_norm[y, x])
        features.append([dx, dy, u, v])

    return np.array(features, dtype=np.float32), h, w


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _match_shape_oriented_numba(
    img_gx: np.ndarray,
    img_gy: np.ndarray,
    img_mag: np.ndarray,
    tpl_features: np.ndarray,
    angles_rad: np.ndarray,
    min_mag: float = 25.0
) -> np.ndarray:
    '''
    Match shape templates over multiple orientations using Numba acceleration.

    Args:
        img_gx (np.ndarray): Normalized horizontal gradient of the scene image.
        img_gy (np.ndarray): Normalized vertical gradient of the scene image.
        img_mag (np.ndarray): Gradient magnitude of the scene image.
        tpl_features (np.ndarray): Shape template features array of shape (N_pts, 4).
        angles_rad (np.ndarray): Array of rotation angles in radians.
        min_mag (float): Minimum gradient magnitude threshold for match validation in the scene.

    Returns:
        np.ndarray: Match score maps of shape (N_angles, H, W).
    '''
    H, W = img_gx.shape
    N_pts = tpl_features.shape[0]
    N_angles = len(angles_rad)

    scores = np.zeros((N_angles, H, W), dtype=np.float32)

    for a_idx in numba.prange(N_angles):
        angle = angles_rad[a_idx]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        rot_pts = np.zeros((N_pts, 4), dtype=np.float32)
        for i in range(N_pts):
            dx, dy, u, v = tpl_features[i, 0], tpl_features[i, 1], tpl_features[i, 2], tpl_features[i, 3]
            rot_pts[i, 0] = dx * cos_a - dy * sin_a
            rot_pts[i, 1] = dx * sin_a + dy * cos_a
            rot_pts[i, 2] = u * cos_a - v * sin_a
            rot_pts[i, 3] = u * sin_a + v * cos_a

        for y in range(H):
            for x in range(W):
                match_score = 0.0

                for i in range(N_pts):
                    px = int(np.round(x + rot_pts[i, 0]))
                    py = int(np.round(y + rot_pts[i, 1]))

                    if 0 <= px < W and 0 <= py < H:
                        if img_mag[py, px] >= min_mag:
                            img_u = img_gx[py, px]
                            img_v = img_gy[py, px]

                            dot = rot_pts[i, 2] * img_u + rot_pts[i, 3] * img_v
                            if dot > 0.0:
                                match_score += dot

                scores[a_idx, y, x] = match_score / float(N_pts)

    return scores


@numba.jit(nopython=True, fastmath=True)
def _nms_shape_3d(
    scores: np.ndarray,
    angles_deg: np.ndarray,
    tpl_h: int,
    tpl_w: int,
    min_score: float = 0.60,
    max_matches: int = 20
) -> np.ndarray:
    '''
    Perform 3D Non-Maximum Suppression (NMS) over position and orientation.

    Args:
        scores (np.ndarray): Score map array of shape (N_angles, H, W).
        angles_deg (np.ndarray): Corresponding angles in degrees.
        tpl_h (int): Height of the template.
        tpl_w (int): Width of the template.
        min_score (float): Minimum matching score threshold.
        max_matches (int): Maximum number of matches to return.

    Returns:
        np.ndarray: Filtered matches array of shape (M, 4), where each row is [cx, cy, angle, score].
    '''
    N_angles, H, W = scores.shape
    candidates = []

    for a in range(N_angles):
        for y in range(H):
            for x in range(W):
                sc = scores[a, y, x]
                if sc >= min_score:
                    candidates.append((sc, float(x), float(y), angles_deg[a]))

    if len(candidates) == 0:
        return np.empty((0, 4), dtype=np.float32)

    n_cand = len(candidates)
    scores_arr = np.zeros(n_cand, dtype=np.float32)
    for i in range(n_cand):
        scores_arr[i] = candidates[i][0]

    sort_idx = np.argsort(-scores_arr)
    suppressed = np.zeros(n_cand, dtype=numba.boolean)

    results = []
    dist_thresh_sq = (min(tpl_h, tpl_w) * 0.2) ** 2

    for i in range(n_cand):
        idx = sort_idx[i]
        if suppressed[idx]:
            continue

        sc, cx, cy, ang = candidates[idx]
        results.append([cx, cy, ang, sc])

        if len(results) >= max_matches:
            break

        for j in range(i + 1, n_cand):
            idx_j = sort_idx[j]
            if suppressed[idx_j]:
                continue

            _, cx_j, cy_j, _ = candidates[idx_j]
            if (cx - cx_j)**2 + (cy - cy_j)**2 < dist_thresh_sq:
                suppressed[idx_j] = True

    res_arr = np.zeros((len(results), 4), dtype=np.float32)
    for i in range(len(results)):
        for j in range(4):
            res_arr[i, j] = results[i][j]

    return res_arr


def apply_shape_matching(
    image: Union[np.ndarray, torch.Tensor],
    template: Union[np.ndarray, torch.Tensor],
    angle_step: float = 2.0,
    min_score: float = 0.60,
    max_matches: int = 20,
    device: str = 'cpu'
) -> np.ndarray:
    '''
    Apply shape-based template matching to find the template occurrences in the image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The scene image.
        template (Union[np.ndarray, torch.Tensor]): The template image.
        angle_step (float): Step size for search angles in degrees.
        min_score (float): Minimum match score threshold.
        max_matches (int): Maximum number of returned matched locations.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        np.ndarray: Detected template instances array of shape (M, 4), where each row is [cx, cy, angle, score].
    '''
    if isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image)
    else:
        img_tensor = image

    if isinstance(template, np.ndarray):
        tpl_tensor = torch.from_numpy(template)
    else:
        tpl_tensor = template

    img_mag, img_gx, img_gy = preprocess_shape_matching_pytorch(img_tensor, device=device)

    tpl_features, tpl_h, tpl_w = extract_template_shape_features(tpl_tensor, mag_thresh=40.0, device=device)

    angles_deg = np.arange(0.0, 360.0, angle_step, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    scores_3d = _match_shape_oriented_numba(img_gx, img_gy, img_mag, tpl_features, angles_rad)

    matches = _nms_shape_3d(scores_3d, angles_deg, tpl_h, tpl_w, min_score=min_score, max_matches=max_matches)

    return matches


def draw_shape_matches_lines(
    ax: Any,
    matches: np.ndarray,
    contour_features: List[np.ndarray],
    tpl_h: int,
    tpl_w: int,
    img_mag: np.ndarray,
    img_gx: np.ndarray,
    img_gy: np.ndarray,
    min_mag: float = 25.0,
    dot_thresh: float = 0.5,
    linewidth: float = 2.0,
    box: bool = True,
) -> None:
    '''
    Draw shape matching results with bounding boxes and contour lines on a matplotlib axis.

    Args:
        ax (Any): Matplotlib axis object to plot on.
        matches (np.ndarray): Matches array of shape (M, 4), where each row is [cx, cy, angle, score].
        contour_features (List[np.ndarray]): List of template contour feature arrays of shape (N, 4) [dx, dy, u, v].
        tpl_h (int): Height of the template image.
        tpl_w (int): Width of the template image.
        img_mag (np.ndarray): Scene gradient magnitude array of shape (H, W).
        img_gx (np.ndarray): Scene normalized horizontal gradient array of shape (H, W).
        img_gy (np.ndarray): Scene normalized vertical gradient array of shape (H, W).
        min_mag (float): Minimum gradient magnitude threshold for match validation in scene.
        dot_thresh (float): Dot product threshold between template gradient and scene gradient for validity.
        linewidth (float): Line width for plotting bounding box and contours.
        box (bool): Whether to draw the rotated bounding box around matches.
    '''
    H, W = img_mag.shape
    box_pts = np.array([
        [-tpl_w / 2.0, -tpl_h / 2.0],
        [ tpl_w / 2.0, -tpl_h / 2.0],
        [ tpl_w / 2.0,  tpl_h / 2.0],
        [-tpl_w / 2.0,  tpl_h / 2.0],
        [-tpl_w / 2.0, -tpl_h / 2.0]
    ], dtype=np.float32)

    for match in matches:
        cx, cy, ang, score = match
        rad = np.deg2rad(ang)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        if box:
            # 1. Draw green rotated bounding box
            rot_box_x = cx + box_pts[:, 0] * cos_a - box_pts[:, 1] * sin_a
            rot_box_y = cy + box_pts[:, 0] * sin_a + box_pts[:, 1] * cos_a
            ax.plot(rot_box_x, rot_box_y, color='#00FF00', linewidth=linewidth)

        # 2. Process ordered contours and split into green/red continuous line segments
        for feat_arr in contour_features:
            N = len(feat_arr)
            if N < 2:
                continue

            scene_x = np.zeros(N, dtype=np.float32)
            scene_y = np.zeros(N, dtype=np.float32)
            is_valid = np.zeros(N, dtype=bool)

            for i in range(N):
                dx, dy, u, v = feat_arr[i]
                px = float(cx + dx * cos_a - dy * sin_a)
                py = float(cy + dx * sin_a + dy * cos_a)

                scene_x[i] = px
                scene_y[i] = py

                ix = int(np.round(px))
                iy = int(np.round(py))

                u_rot = u * cos_a - v * sin_a
                v_rot = u * sin_a + v * cos_a

                if 0 <= ix < W and 0 <= iy < H:
                    if img_mag[iy, ix] >= min_mag:
                        dot = u_rot * img_gx[iy, ix] + v_rot * img_gy[iy, ix]
                        if dot >= dot_thresh:
                            is_valid[i] = True

            # Split contiguous points into line segments and plot
            curr_seg_x = [scene_x[0]]
            curr_seg_y = [scene_y[0]]
            curr_status = is_valid[0]

            for i in range(1, N):
                status = is_valid[i]
                if status == curr_status:
                    curr_seg_x.append(scene_x[i])
                    curr_seg_y.append(scene_y[i])
                else:
                    curr_seg_x.append(scene_x[i])
                    curr_seg_y.append(scene_y[i])

                    color_str = '#00FF00' if curr_status else '#FF0000'
                    ax.plot(curr_seg_x, curr_seg_y, color=color_str, linewidth=linewidth)

                    curr_seg_x = [scene_x[i]]
                    curr_seg_y = [scene_y[i]]
                    curr_status = status

            if len(curr_seg_x) > 1:
                color_str = '#00FF00' if curr_status else '#FF0000'
                ax.plot(curr_seg_x, curr_seg_y, color=color_str, linewidth=linewidth)

        # 3. Draw center marker & match score
        ax.plot(cx, cy, 'ro', markersize=4)
        ax.text(cx - 20, cy - 25, f"{score:.3f}", color='blue', fontsize=13, weight='bold')


def draw_part_smooth(
    scene: np.ndarray,
    template: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float
) -> None:
    '''
    Draw a rotated template onto a scene image using bilinear interpolation.

    Args:
        scene (np.ndarray): Scene image to modify in place.
        template (np.ndarray): Template image.
        cx (float): X-coordinate of the center of rotation on the scene.
        cy (float): Y-coordinate of the center of rotation on the scene.
        angle_deg (float): Rotation angle in degrees.
    '''
    h, w = template.shape
    H, W = scene.shape
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    radius = int(np.ceil(np.sqrt(h*h + w*w) / 2.0))
    x_min, x_max = max(0, int(cx - radius)), min(W, int(cx + radius + 1))
    y_min, y_max = max(0, int(cy - radius)), min(H, int(cy + radius + 1))

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dx, dy = x - cx, y - cy
            tx =  dx * cos_a + dy * sin_a + w / 2.0
            ty = -dx * sin_a + dy * cos_a + h / 2.0

            if 0 <= tx < w - 1 and 0 <= ty < h - 1:
                x0, y0 = int(tx), int(ty)
                fx, fy = tx - x0, ty - y0

                val = (1-fx)*(1-fy)*template[y0, x0] + \
                      fx*(1-fy)*template[y0, x0+1] + \
                      (1-fx)*fy*template[y0+1, x0] + \
                      fx*fy*template[y0+1, x0+1]

                if val > 10.0:
                    scene[y, x] = max(scene[y, x], val)
