import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
import matplotlib.pyplot as plt

from codon import *
from codon.ops.cv.matching.shape import (
    preprocess_shape_matching_pytorch,
    extract_template_ordered_contours,
    apply_shape_matching,
    draw_shape_matches_lines,
    draw_part_smooth
)


def generate_complex_gear_template(radius: int = 45, num_teeth: int = 7) -> np.ndarray:
    size = radius * 2 + 30
    cx, cy = size // 2, size // 2
    template = np.zeros((size, size), dtype=np.float32)

    y, x = np.ogrid[:size, :size]
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    teeth_pattern = radius + 8.0 * np.sin(num_teeth * angle)
    gear_mask = (dist <= teeth_pattern) & (dist >= radius * 0.4)
    template[gear_mask] = 220.0

    hole_mask = dist < radius * 0.25
    template[hole_mask] = 0.0

    notch_x = cx + int(radius * 0.6 * np.cos(np.pi / 4))
    notch_y = cy + int(radius * 0.6 * np.sin(np.pi / 4))
    notch_dist = np.sqrt((x - notch_x)**2 + (y - notch_y)**2)
    template[notch_dist < 4.0] = 0.0

    return template


def build_overlapping_complex_scene(
    template: np.ndarray,
    scene_shape: Tuple[int, int] = (650, 900)
) -> np.ndarray:
    H, W = scene_shape
    scene = np.zeros((H, W), dtype=np.float32)

    y, x = np.ogrid[:H, :W]
    background = 25.0 + 35.0 * (x / W) + 30.0 * np.sin(y / 40.0) + 15.0 * np.cos(x / 60.0)
    scene += background

    targets = [
        (220.0, 200.0, 25.0),
        (265.0, 235.0, 140.0),
        
        (580.0, 300.0, 210.0),
        (630.0, 260.0, 45.0),
        (640.0, 330.0, 310.0),

        (320.0, 480.0, 95.0),
        (750.0, 480.0, 175.0),
    ]

    for cx, cy, angle in targets:
        draw_part_smooth(scene, template, cx, cy, angle)

    np.random.seed(42)
    
    for _ in range(20):
        x1, y1 = np.random.randint(0, W), np.random.randint(0, H)
        x2, y2 = np.random.randint(0, W), np.random.randint(0, H)
        rr, cc = np.linspace(y1, y2, 120).astype(int), np.linspace(x1, x2, 120).astype(int)
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        scene[rr[valid], cc[valid]] = 200.0

    for _ in range(6):
        rcx, rcy = np.random.randint(50, W - 50), np.random.randint(50, H - 50)
        r_rad = np.random.randint(15, 60)
        c_dist = np.sqrt((x - rcx)**2 + (y - rcy)**2)
        scene[np.abs(c_dist - r_rad) < 2.0] = 170.0

    noise = np.random.normal(0.0, 16.0, (H, W)).astype(np.float32)
    scene = np.clip(scene + noise, 0.0, 255.0)

    return scene


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    template_np = generate_complex_gear_template(radius=45, num_teeth=7)
    template_tensor = torch.from_numpy(template_np)

    contour_features, tpl_h, tpl_w = extract_template_ordered_contours(
        template_tensor, mag_thresh=35.0, device=device
    )

    scene_np = build_overlapping_complex_scene(template_np, scene_shape=(650, 900))
    scene_tensor = torch.from_numpy(scene_np)

    t0 = time.time()
    matches = apply_shape_matching(
        image=scene_tensor,
        template=template_tensor,
        angle_step=2.0,
        min_score=0.48,
        max_matches=12,
        device=device
    )
    match_time = (time.time() - t0) * 1000.0
    print(f"[+] Matching done in {match_time:.2f} ms. Detected instances: {len(matches)}")

    img_mag, img_gx, img_gy = preprocess_shape_matching_pytorch(scene_tensor, device=device)

    plt.figure(figsize=(14, 9))
    ax = plt.gca()
    ax.imshow(scene_np, cmap='gray')
    ax.set_title(
        f"Shape Matching Result (Overlapping & Occlusion Scene) - Found {len(matches)} Targets",
        fontsize=14,
        weight='bold'
    )

    draw_shape_matches_lines(
        ax=ax,
        matches=matches,
        contour_features=contour_features,
        tpl_h=tpl_h,
        tpl_w=tpl_w,
        img_mag=img_mag,
        img_gx=img_gx,
        img_gy=img_gy,
        min_mag=20.0,
        dot_thresh=0.40,
        linewidth=1.8,
        box=False
    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()