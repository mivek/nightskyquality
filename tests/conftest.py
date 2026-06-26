import numpy as np
import rasterio


def make_synthetic_raster(path, *, height, width, epsg, resolution_m, value=0.0, dtype='float64'):
    """Create a GeoTIFF with uniform value at given path."""
    transform = rasterio.transform.from_origin(0, height * resolution_m, resolution_m, resolution_m)
    arr = np.full((height, width), value, dtype=dtype)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width, count=1,
        dtype=dtype, crs=f'EPSG:{epsg}', transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)
    return path


def point_source_raster(tmp_path):
    """1000x1000, EPSG:3035, 450m, single 100 nW/cm²/sr pixel at center."""
    path = tmp_path / 'point_source.tif'
    arr = np.zeros((1000, 1000), dtype=np.float64)
    arr[500, 500] = 100.0
    transform = rasterio.transform.from_origin(0, 1000 * 450, 450, 450)
    with rasterio.open(
        path, 'w', driver='GTiff', height=1000, width=1000, count=1,
        dtype='float64', crs='EPSG:3035', transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(arr, 1)
    return path

