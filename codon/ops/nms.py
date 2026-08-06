from codon import *

@numba.jit(nopython=True, fastmath=True)
def nms_2d_suppression(
    scores: np.ndarray, 
    box_h: int, 
    box_w: int, 
    threshold: float = 0.5, 
    max_matches: int = 100
) -> np.ndarray:
    H, W = scores.shape
    candidates = []
    for y in range(H):
        for x in range(W):
            if scores[y, x] >= threshold:
                candidates.append((scores[y, x], x, y))

    if len(candidates) == 0:
        return np.empty((0, 5), dtype=np.float32)

    n_cand = len(candidates)
    scores_arr = np.array([c[0] for c in candidates], dtype=np.float32)
    sort_indices = np.argsort(-scores_arr)

    suppressed = np.zeros(n_cand, dtype=numba.boolean)
    results = []

    half_w, half_h = box_w / 2.0, box_h / 2.0

    for i in range(n_cand):
        idx = sort_indices[i]
        if suppressed[idx]:
            continue

        sc, cx, cy = candidates[idx]
        results.append([float(cx), float(cy), float(cx + box_w), float(cy + box_h), sc])
        
        if len(results) >= max_matches:
            break

        for j in range(i + 1, n_cand):
            idx_j = sort_indices[j]
            if suppressed[idx_j]:
                continue
            cx_j, cy_j = candidates[idx_j][1], candidates[idx_j][2]
            if abs(cx - cx_j) < half_w and abs(cy - cy_j) < half_h:
                suppressed[idx_j] = True

    return np.array(results, dtype=np.float32)

@numba.jit(nopython=True, fastmath=True)
def nms_2d_peaks(
    response_map: np.ndarray,
    h: int,
    w: int,
    threshold: float = 0.8,
    max_matches: int = 100,
    is_sqdiff: bool = False
) -> np.ndarray:
    '''2D 矩形/响应图 NMS 算子 (提取自 template.py)'''
    H_map, W_map = response_map.shape
    candidates = []

    for y in range(H_map):
        for x in range(W_map):
            score = response_map[y, x]
            if is_sqdiff and score <= threshold:
                candidates.append((score, x, y))
            elif not is_sqdiff and score >= threshold:
                candidates.append((score, x, y))

    if len(candidates) == 0:
        return np.empty((0, 5), dtype=np.float32)

    n_cand = len(candidates)
    scores = np.array([c[0] for c in candidates], dtype=np.float32)
    xs = np.array([c[1] for c in candidates], dtype=np.int32)
    ys = np.array([c[2] for c in candidates], dtype=np.int32)

    sort_indices = np.argsort(scores) if is_sqdiff else np.argsort(-scores)
    suppressed = np.zeros(n_cand, dtype=numba.boolean)
    boxes = []
    half_w, half_h = w / 2.0, h / 2.0

    for i in range(n_cand):
        idx = sort_indices[i]
        if suppressed[idx]:
            continue

        cx, cy, sc = xs[idx], ys[idx], scores[idx]
        boxes.append([float(cx), float(cy), float(cx + w), float(cy + h), sc])
        if len(boxes) >= max_matches:
            break

        for j in range(i + 1, n_cand):
            idx_j = sort_indices[j]
            if not suppressed[idx_j] and abs(cx - xs[idx_j]) < half_w and abs(cy - ys[idx_j]) < half_h:
                suppressed[idx_j] = True

    res = np.zeros((len(boxes), 5), dtype=np.float32)
    for i in range(len(boxes)):
        for j in range(5):
            res[i, j] = boxes[i][j]
    return res
