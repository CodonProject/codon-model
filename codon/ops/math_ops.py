from codon import *


@numba.jit(nopython=True, fastmath=True)
def angle_diff(a1: float, a2: float) -> float:
    '''
    Calculate absolute difference between two angles in radians.

    Args:
        a1 (float): Angle 1 in radians.
        a2 (float): Angle 2 in radians.

    Returns:
        float: Absolute difference in radians.
    '''
    diff = np.abs(a1 - a2)
    if diff > np.pi:
        diff = 2.0 * np.pi - diff
    if diff > np.pi / 2.0:
        diff = np.pi - diff
    return diff

@numba.jit(nopython=True, fastmath=True)
def solve_3x3(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, bool]:
    '''
    Solve 3x3 linear system Ax = b using Cramer's rule.

    Args:
        A (np.ndarray): 3x3 coefficient matrix.
        b (np.ndarray): 3-element right hand side vector.

    Returns:
        Tuple[np.ndarray, bool]: A tuple containing the solution vector and a boolean success flag.
    '''
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
    a20, a21, a22 = A[2, 0], A[2, 1], A[2, 2]
    b0, b1, b2    = b[0],    b[1],    b[2]

    detA = (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20))
    if abs(detA) < 1e-10:
        return np.zeros(3, dtype=np.float32), False

    detX = (b0  * (a11 * a22 - a12 * a21) - a01 * (b1  * a22 - a12 * b2)  + a02 * (b1  * a21 - a11 * b2))
    detY = (a00 * (b1  * a22 - a12 * b2)  - b0  * (a10 * a22 - a12 * a20) + a02 * (a10 * b2  - b1  * a20))
    detZ = (a00 * (a11 * b2  - b1  * a21) - a01 * (a10 * b2  - b1  * a20) + b0  * (a10 * a21 - a11 * a20))

    return np.array([detX / detA, detY / detA, detZ / detA], dtype=np.float32), True

@numba.jit(nopython=True, fastmath=True)
def l2_hys_normalize(vec: np.ndarray, eps: float = 1e-5, max_val: float = 0.2) -> np.ndarray:
    '''
    Normalize a vector using L2-Hys (L2-Hysteresis).

    Args:
        vec (np.ndarray): Input 1D array/vector.
        eps (float): Stability epsilon. Defaults to 1e-5.
        max_val (float): Max value for clipping. Defaults to 0.2.

    Returns:
        np.ndarray: Normalized 1D array.
    '''
    norm1 = np.sqrt(np.sum(vec**2) + eps**2)
    vec = np.minimum(vec / norm1, max_val)
    norm2 = np.sqrt(np.sum(vec**2) + eps**2)
    return vec / norm2
