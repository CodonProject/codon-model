'''
Template and shape matching operations subpackage.
'''

from .shape import (
    preprocess_shape_matching_pytorch,
    extract_template_shape_features,
    apply_shape_matching,
    draw_part_smooth,
    draw_shape_matches_lines
)
from .template import (
    compute_template_matching_map,
    apply_template_matching,
)

__all__ = [
    'preprocess_shape_matching_pytorch',
    'extract_template_shape_features',
    'apply_shape_matching',
    'draw_shape_matches_lines',
    'draw_part_smooth',
    'compute_template_matching_map',
    'apply_template_matching',
]
