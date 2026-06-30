import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    """Read band 1. Return (float64 array, rasterio profile dict)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    return arr, profile


def write_geotiff(path: str, arr: np.ndarray, profile: dict) -> None:
    """Write GeoTIFF with tiled+DEFLATE settings.

    This is the recommended way to persist an ALRResult (tiled, DEFLATE-compressed
    GeoTIFF with float64 dtype and NaN nodata). Use it to write ALRResult.data
    to disk with the same tuned profile settings used internally.
    """
    out_profile = profile.copy()
    out_profile.update(
        driver='GTiff',
        dtype='float64',
        nodata=np.nan,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress='deflate',
        predictor=2,
        bigtiff='if_safer',
    )
    out_profile['count'] = 1
    with rasterio.open(path, 'w', **out_profile) as dst:
        dst.write(arr, 1)


def reproject_raster(
    src_arr: np.ndarray,
    src_profile: dict,
    dst_epsg: int,
    dst_resolution: float,
) -> tuple[np.ndarray, dict]:
    """Reproject to target CRS at target resolution. Returns (array, new_profile).

    EC-11: no-op if CRS matches (returns same array object).
    """
    dst_crs_str = f'EPSG:{dst_epsg}'
    src_crs = src_profile['crs']
    if src_crs is not None:
        # Handle both CRS objects and strings
        src_epsg = src_crs.to_epsg() if hasattr(src_crs, 'to_epsg') else None
        if src_epsg == dst_epsg or str(src_crs) == dst_crs_str:
            return src_arr, src_profile
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs_str, src_profile['width'], src_profile['height'],
        left=src_profile['transform'][2],
        bottom=src_profile['transform'][5] - src_profile['height'] * abs(src_profile['transform'][4]),
        right=src_profile['transform'][2] + src_profile['width'] * src_profile['transform'][0],
        top=src_profile['transform'][5],
        resolution=dst_resolution,
    )
    dst_arr = np.zeros((height, width), dtype=np.float64)
    reproject(
        source=src_arr, destination=dst_arr,
        src_transform=src_profile['transform'], src_crs=src_crs,
        dst_transform=transform, dst_crs=dst_crs_str,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    dst_profile = src_profile.copy()
    dst_profile.update(crs=dst_crs_str, transform=transform, width=width, height=height)
    return dst_arr, dst_profile


def apply_preconvolution_mask(arr: np.ndarray, profile: dict, noise_floor: float) -> np.ndarray:
    """Apply masking BEFORE convolution: NaN→0, nodata→0, negative→0, <noise_floor→0.

    Returns float64. Does NOT modify input array.
    """
    result = arr.copy()
    result[np.isnan(result)] = 0.0
    if profile and 'nodata' in profile and profile['nodata'] is not None:
        ndv = profile['nodata']
        if not np.isnan(ndv):
            result[result == ndv] = 0.0
    result[result < 0] = 0.0
    result[result < noise_floor] = 0.0
    return result
