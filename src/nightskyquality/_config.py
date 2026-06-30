from dataclasses import dataclass


@dataclass(frozen=True)
class ALRConfig:
    work_resolution_m: int = 450
    rings: int = 38
    max_km: int = 300
    alpha_base: float = 2.3
    alpha_exp: float = 0.28
    calib_c: float = 1.0 / 562.72
    noise_floor: float = 0.5
    eps: float = 0.001
