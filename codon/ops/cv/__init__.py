'''
Computer vision operations subpackage.

This package contains implementations of various classical computer vision
algorithms including Canny edge detection, ELSD/LSD line/ellipse detection,
HOG descriptor extraction, Hough transform, SIFT feature extraction, and SLIC
superpixel segmentation.
'''

from .canny import apply_canny, preprocess_canny_pytorch
from .elsd import apply_elsd, apply_lsd, lsd_core_numba, elsd_core_numba
from .hog import apply_hog, preprocess_hog_pytorch
from .hough import apply_hough, preprocess_hough_pytorch, hough_lines_to_endpoints
from .sift import apply_sift, build_pyramids_pytorch
from .slic import apply_slic, find_boundaries, preprocess_slic_pytorch

__all__ = [
    'apply_canny',
    'preprocess_canny_pytorch',
    'apply_elsd',
    'apply_lsd',
    'lsd_core_numba',
    'elsd_core_numba',
    'apply_hog',
    'preprocess_hog_pytorch',
    'apply_hough',
    'preprocess_hough_pytorch',
    'hough_lines_to_endpoints',
    'apply_sift',
    'build_pyramids_pytorch',
    'apply_slic',
    'find_boundaries',
    'preprocess_slic_pytorch',
]
