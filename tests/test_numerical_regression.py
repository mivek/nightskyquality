import numpy as np
from scipy.signal import fftconvolve, convolve2d
from nightskyquality._kernel import master_kernel
from nightskyquality._config import ALRConfig


def test_fft_vs_spatial_same_kernel():
    """FFT convolution must match spatial convolution on the same master kernel."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0, 10, (80, 80)).astype(np.float64)

    cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)
    K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                      cfg.alpha_base, cfg.alpha_exp, cfg.calib_c)

    fft_result = fftconvolve(arr, K, mode='same')
    spatial_result = convolve2d(arr, K, mode='same')

    assert np.allclose(fft_result, spatial_result, rtol=1e-10, atol=1e-12), \
        f"Max diff: {np.max(np.abs(fft_result - spatial_result))}"
