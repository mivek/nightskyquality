import numpy as np
from ._weights import ring_boundaries_px, ring_midpoints_km, ring_weights


def annulus_kernel(r_in: float, r_out: int) -> np.ndarray:
    """Binary ring kernel: 1 inside annulus [r_in, r_out], 0 outside.

    Shape: (2*r_out+1, 2*r_out+1). Center pixel excluded by r_in (eps).
    Uses Euclidean distance — NO astropy Ellipse2D.
    """
    size = 2 * r_out + 1
    y, x = np.ogrid[-r_out:r_out+1, -r_out:r_out+1]
    dist = np.sqrt(x*x + y*y)
    kernel = ((dist >= r_in) & (dist <= r_out)).astype(np.float64)
    return kernel


def master_kernel(
    rings: int, max_km: int, resolution_m: int,
    alpha_base: float, alpha_exp: float, calib_c: float,
    eps: float = 0.001,
) -> np.ndarray:
    """Precompute K = C · Σ w_i · annulus_kernel(b_i+eps, b_{i+1}).

    Returns float64, shape (2*R_px+1, 2*R_px+1) where R_px = max_px.
    Calibration constant C folded INSIDE the kernel.
    """
    boundaries_px = ring_boundaries_px(max_km, resolution_m, rings)
    midpoints_km = ring_midpoints_km(boundaries_px, resolution_m)
    weights = ring_weights(midpoints_km, alpha_base, alpha_exp)
    r_out_max = int(boundaries_px[-1])  # 666 at 450m/300km
    size = 2 * r_out_max + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    # Precompute coordinate grid for the full kernel
    y, x = np.ogrid[-r_out_max:r_out_max+1, -r_out_max:r_out_max+1]
    dist = np.sqrt(x*x + y*y)
    for i in range(rings):
        r_in = float(boundaries_px[i]) + eps
        r_out = int(boundaries_px[i + 1])
        ann = ((dist >= r_in) & (dist <= r_out)).astype(np.float64)
        kernel += weights[i] * ann
    kernel *= calib_c
    return kernel
