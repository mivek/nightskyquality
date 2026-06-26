import numpy as np


def alpha(d_km: float, base: float = 2.3, exp: float = 0.28) -> float:
    """α(d) = base · (d / 350)^exp"""
    return base * (d_km / 350.0) ** exp


def ring_boundaries_px(max_km: float, resolution_m: float, rings: int) -> np.ndarray:
    """Annulus ring cutoffs in pixel units. Returns length rings+1 int array.

    MATCHES ORIGINAL: np.linspace(0, dist2cell(max_km), rings+1, dtype=int)
    where dist2cell(x) = x * 1000 / resolution_m (FLOORED to int).

    Note: numpy.linspace with dtype=int truncates intermediate values, so
    rings are NOT perfectly evenly spaced in pixel units (adjacent ring
    widths differ by ±1 px). This matches the original script behavior.
    """
    max_px = int(max_km * 1000 / resolution_m)  # floors (matches original)
    return np.linspace(0, max_px, rings + 1, dtype=int)


def ring_midpoints_km(boundaries_px: np.ndarray, resolution_m: float) -> np.ndarray:
    """Midpoint distances in km for each ring. Length = len(boundaries_px)-1.

    MATCHES ORIGINAL: np.convolve(cell2dist(boundaries), [0.5,0.5], mode='valid')
    """
    dist_km = boundaries_px * resolution_m / 1000.0
    return np.convolve(dist_km, np.ones(2) / 2, mode='valid')


def ring_weights(midpoints_km: np.ndarray, alpha_base: float, alpha_exp: float) -> np.ndarray:
    """w_i = d_i^(-α(d_i)). Returns float64 array."""
    return np.array([d ** (-alpha(d, alpha_base, alpha_exp)) for d in midpoints_km], dtype=np.float64)
