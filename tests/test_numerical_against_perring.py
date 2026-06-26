import numpy as np
from scipy.signal import fftconvolve
from nightskyquality._kernel import master_kernel, annulus_kernel
from nightskyquality._weights import ring_boundaries_px, ring_midpoints_km, ring_weights
from nightskyquality._config import ALRConfig


def test_master_vs_per_ring_sum():
    """Linearity: Σ w_i·(img ⊛ k_i) ≡ img ⊛ (Σ w_i·k_i)."""
    cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)

    rng = np.random.default_rng(42)
    arr = rng.uniform(0, 10, (80, 80)).astype(np.float64)

    boundaries_px = ring_boundaries_px(cfg.max_km, cfg.work_resolution_m, cfg.rings)
    midpoints_km = ring_midpoints_km(boundaries_px, cfg.work_resolution_m)
    weights = ring_weights(midpoints_km, cfg.alpha_base, cfg.alpha_exp)

    per_ring_sum = np.zeros_like(arr)
    for i in range(cfg.rings):
        r_in = float(boundaries_px[i]) + cfg.eps
        r_out = int(boundaries_px[i+1])
        k_i = annulus_kernel(r_in, r_out)
        per_ring_sum += weights[i] * fftconvolve(arr, k_i, mode='same')
    per_ring_sum *= cfg.calib_c

    K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                      cfg.alpha_base, cfg.alpha_exp, cfg.calib_c, cfg.eps)
    master_result = fftconvolve(arr, K, mode='same')

    assert np.allclose(master_result, per_ring_sum, atol=1e-8), \
        f"Max diff: {np.max(np.abs(master_result - per_ring_sum))}"
