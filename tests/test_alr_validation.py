import numpy as np
import pytest
from nightskyquality import radiance_to_alr
from tests.conftest import point_source_raster


# ── Parameter validation ──────────────────────────────

def test_rings_too_small(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError, match="rings must be >= 2"):
        radiance_to_alr(str(path), equal_area_epsg=3035, rings=1, max_km=50)

def test_rings_zero(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError):
        radiance_to_alr(str(path), equal_area_epsg=3035, rings=0, max_km=50)

def test_max_km_too_small(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError):
        radiance_to_alr(str(path), equal_area_epsg=3035, max_km=0.1)

def test_calib_c_zero(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError):
        radiance_to_alr(str(path), equal_area_epsg=3035, max_km=50, calib_c=0)

def test_calib_c_negative(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError):
        radiance_to_alr(str(path), equal_area_epsg=3035, max_km=50, calib_c=-1)

def test_noise_floor_negative(tmp_path):
    path = point_source_raster(tmp_path)
    with pytest.raises(ValueError):
        radiance_to_alr(str(path), equal_area_epsg=3035, max_km=50, noise_floor=-1)

def test_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        radiance_to_alr("/nonexistent/file.tif", equal_area_epsg=3035)


# ── Edge-NaN logic ─────────────────────────────────────

def test_edge_nan_band(tmp_path):
    """Outer R_px of output is NaN; inner region is non-NaN."""
    path = point_source_raster(tmp_path)
    result = radiance_to_alr(str(path), equal_area_epsg=3035, max_km=50)
    alr = result.data
    R_px = int(50 * 1000 / 450)  # 111

    assert np.isnan(alr[0, 0])                  # corner is in edge band
    assert not np.isnan(alr[500, 500])          # center is valid

    # Full outer strips are NaN
    assert np.all(np.isnan(alr[:R_px, :]))
    assert np.all(np.isnan(alr[-R_px:, :]))
    assert np.all(np.isnan(alr[:, :R_px]))
    assert np.all(np.isnan(alr[:, -R_px:]))
