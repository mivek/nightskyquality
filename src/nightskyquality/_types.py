from dataclasses import dataclass
import numpy as np


@dataclass
class ALRResult:
    data: np.ndarray  # 2D float64 (H,W), NaN at outer R_px
    profile: dict  # rasterio profile dict


@dataclass(frozen=True)
class TileSpec:
    read_window: tuple[int, int, int, int]  # (row_start, row_end, col_start, col_end) INCLUSIVE
    valid_slice: tuple[slice, slice]  # slice into convolved read window
