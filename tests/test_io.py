import numpy as np
import rasterio
from nightskyquality._io import apply_preconvolution_mask, read_raster, write_geotiff, reproject_raster


class TestApplyPreconvolutionMask:
    def test_nan_to_zero(self):
        arr = np.array([[1.0, np.nan, 2.0]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {}, noise_floor=0.0)
        assert result[0, 1] == 0.0
        assert result[0, 0] == 1.0

    def test_nodata_tag_present(self):
        arr = np.array([[1.0, -999.0, 3.0]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {'nodata': -999.0}, noise_floor=0.0)
        assert result[0, 1] == 0.0
        assert result[0, 0] == 1.0

    def test_nodata_tag_absent_no_masking(self):
        """Without nodata tag, sentinel value is treated as valid data.
        Use a positive sentinel to avoid the negative-clamping rule."""
        arr = np.array([[1.0, 999.0, 3.0]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {}, noise_floor=0.0)
        assert result[0, 1] == 999.0

    def test_nodata_tag_none(self):
        arr = np.array([[1.0, 5.0]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {'nodata': None}, noise_floor=0.0)
        np.testing.assert_array_equal(result, arr)

    def test_negative_to_zero(self):
        arr = np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {}, noise_floor=0.0)
        assert result[0, 0] == 0.0

    def test_below_noise_floor_to_zero(self):
        """Values strictly below noise_floor are zeroed. Values at or above remain."""
        arr = np.array([[0.1, 0.5, 0.6]], dtype=np.float64)
        result = apply_preconvolution_mask(arr, {}, noise_floor=0.5)
        assert result[0, 0] == 0.0     # 0.1 < 0.5 → zeroed
        assert result[0, 1] == 0.5     # 0.5 == noise_floor → unchanged (strict <)
        assert result[0, 2] == 0.6     # above → unchanged

    def test_input_not_mutated(self):
        arr = np.array([[np.nan, -1.0, -999.0, 0.3, 5.0]], dtype=np.float64)
        original = arr.copy()
        _ = apply_preconvolution_mask(arr, {'nodata': -999.0}, noise_floor=0.5)
        np.testing.assert_array_equal(arr, original)


class TestReadWriteRoundTrip:
    def test_roundtrip_and_geotiff_profile(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float64)
        profile = {'driver': 'GTiff', 'height': 2, 'width': 2, 'count': 1,
                   'dtype': 'float64', 'crs': 'EPSG:4326',
                   'transform': rasterio.transform.from_origin(0, 2, 1, 1)}
        path = tmp_path / 'test.tif'
        write_geotiff(str(path), arr, profile)
        result, out_profile = read_raster(str(path))
        # NaN-aware equality
        valid = ~np.isnan(arr)
        assert np.allclose(result[valid], arr[valid], equal_nan=True)
        assert np.all(np.isnan(result) == np.isnan(arr))
        # Output profile asserts
        assert out_profile['tiled'] is True
        assert out_profile['compress'] == 'deflate'
        assert out_profile['blockxsize'] == 256
        assert out_profile['blockysize'] == 256
        assert np.isnan(out_profile['nodata'])


class TestReprojectRaster:
    def test_noop_when_same_epsg(self):
        arr = np.ones((10, 10), dtype=np.float64)
        profile = {'crs': 'EPSG:3035',
                   'transform': rasterio.transform.from_origin(0, 10, 1, 1),
                   'width': 10, 'height': 10, 'driver': 'GTiff', 'dtype': 'float64'}
        result, out_prof = reproject_raster(arr, profile, 3035, 1.0)
        assert result is arr   # same object returned — no copy
        assert out_prof is profile

    def test_actual_reproject_changes_shape_and_crs(self):
        arr = np.ones((10, 10), dtype=np.float64)
        profile = {'crs': 'EPSG:3035',
                   'transform': rasterio.transform.from_origin(0, 10, 1, 1),
                   'width': 10, 'height': 10, 'driver': 'GTiff', 'dtype': 'float64'}
        result, out_prof = reproject_raster(arr, profile, 4326, 0.01)
        # CRS check: handle both CRS objects and strings
        crs_val = out_prof['crs']
        if hasattr(crs_val, 'to_epsg'):
            assert crs_val.to_epsg() == 4326
        else:
            assert '4326' in str(crs_val)
        assert result.shape[0] > 0 and result.shape[1] > 0
