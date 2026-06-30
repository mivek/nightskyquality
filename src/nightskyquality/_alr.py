import gc
import os
import numpy as np
from scipy.signal import fftconvolve
from rasterio.warp import reproject, Resampling
from ._kernel import master_kernel
from ._tiling import tile_grid, assemble_mosaic
from ._io import read_raster, reproject_raster, apply_preconvolution_mask
from ._types import ALRResult


def radiance_to_alr(
    radiance_path: str,
    equal_area_epsg: int,
    *,
    work_resolution_m: int = 450,
    rings: int = 38,
    max_km: int = 300,
    alpha_base: float = 2.3,
    alpha_exp: float = 0.28,
    calib_c: float = 1 / 562.72,
    noise_floor: float = 0.5,
) -> ALRResult:
    # ── Validation ──────────────────────────────────────
    if not os.path.isfile(radiance_path):
        raise FileNotFoundError(f"Input raster not found: {radiance_path}")
    if rings < 2:
        raise ValueError(f"rings must be >= 2, got {rings}")
    if max_km <= work_resolution_m / 1000:
        raise ValueError(f"max_km ({max_km}) must be > resolution_km ({work_resolution_m/1000})")
    if calib_c <= 0:
        raise ValueError(f"calib_c must be > 0, got {calib_c}")
    if noise_floor < 0:
        raise ValueError(f"noise_floor must be >= 0, got {noise_floor}")

    # ── Read input & capture original metadata ──────────
    _arr, input_profile = read_raster(radiance_path)
    original_crs = input_profile['crs']
    original_transform = input_profile['transform']
    original_width = input_profile['width']
    original_height = input_profile['height']

    # ── Reproject to equal-area ─────────────────────────
    arr, ea_profile = reproject_raster(_arr, input_profile, equal_area_epsg, work_resolution_m)
    del _arr

    # ── Pre-convolution masking ─────────────────────────
    arr = apply_preconvolution_mask(arr, ea_profile, noise_floor)

    # ── Build master kernel ─────────────────────────────
    R_px = int(max_km * 1000 / work_resolution_m)  # 666
    K = master_kernel(rings, max_km, work_resolution_m, alpha_base, alpha_exp, calib_c)

    # ── Tile → convolve → assemble ──────────────────────
    H, W = arr.shape
    tile_size = 2000  # internal default, not configurable
    specs = tile_grid(H, W, tile_size, R_px)
    tiles_alr = []

    for spec in specs:
        r0, r1, c0, c1 = spec.read_window
        tile = arr[r0:r1, c0:c1].copy()
        tile_masked = apply_preconvolution_mask(tile, ea_profile, noise_floor)
        alr_tile = fftconvolve(tile_masked, K, mode='same').astype(np.float64)
        tiles_alr.append(alr_tile)
        del tile          # free source slice; alr_tile retained in tiles_alr list
        gc.collect()

    alr_ea = assemble_mosaic(tiles_alr, specs, (H, W))

    # ── Edge NaN (outer R_px) ───────────────────────────
    alr_ea[:R_px, :] = np.nan
    alr_ea[-R_px:, :] = np.nan
    alr_ea[:, :R_px] = np.nan
    alr_ea[:, -R_px:] = np.nan

    # ── Reproject back to original CRS (pinned approach) ─
    # Always reproject back to input's original CRS (no toggle).
    out_arr = np.full((original_height, original_width), np.nan, dtype=np.float64)
    reproject(
        source=alr_ea,
        destination=out_arr,
        src_transform=ea_profile['transform'],
        src_crs=ea_profile['crs'],
        dst_transform=original_transform,
        dst_crs=original_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    out_profile = input_profile.copy()
    out_profile.update(dtype='float64', nodata=np.nan)

    return ALRResult(data=out_arr, profile=out_profile)
