"""Geographic validation: Paris as a Bortle 8-9 reference.

This test is SKIPPED by default because it requires a real VIIRS radiance
GeoTIFF covering Paris. To enable:

  1. Download a VIIRS monthly composite (e.g., from EOG:
     https://eogdata.mines.edu/products/vnl/) for a tile covering Paris.
  2. Save it as a GeoTIFF (e.g., using rasterio or gdalwarp).
  3. Set the environment variable ALR_PARIS_RADIANCE to the file path.
  4. Run: pytest tests/test_geographic_validation.py -v

The test converts ALR → Bortle using a heuristic placeholder mapping
and asserts the class is 8-9 (Paris is one of the most light-polluted
cities in Europe). The thresholds are calibration scaffolding per
pipeline spec §6 and MUST be empirically tuned.

NOTE: No published ALR-to-Bortle conversion exists. Duriscoe 2018
defines the ALR formula, not a Bortle lookup. The Bortle scale
(darksky.org) uses naked-eye limiting magnitude and SQM (mag/arcsec²).
"""
import os
import numpy as np
import pytest
import rasterio.transform
from nightskyquality import radiance_to_alr


# Approximate Paris center in WGS84 (lon, lat)
PARIS_LON, PARIS_LAT = 2.3522, 48.8566


def alr_to_bortle(alr: float) -> int:
    """Heuristic ALR → Bortle class mapping (calibration scaffolding).

    NOTE: This is a PLACEHOLDER mapping, not a published conversion. The Duriscoe
    2018 paper defines the ALR formula and US calibration constant — it does NOT
    publish an ALR-to-Bortle class lookup. The standard Bortle scale (darksky.org)
    is defined in terms of naked-eye limiting magnitude and SQM (mag/arcsec²),
    not ALR. These thresholds are a log-scale approximation for calibration
    scaffolding only and MUST be empirically tuned per pipeline spec §6 using
    lightpollutionmap (Sky Brightness layer) as ground truth on European
    control points. Do NOT use this function in production for absolute Bortle
    classification without empirical validation.
    """
    if alr < 0.1:
        return 1  # Excellent dark sky
    elif alr < 0.3:
        return 2  # Typical truly dark site
    elif alr < 1.0:
        return 3  # Rural sky
    elif alr < 3.0:
        return 4  # Rural/suburban transition
    elif alr < 10.0:
        return 5  # Suburban sky
    elif alr < 30.0:
        return 6  # Bright suburban sky
    elif alr < 100.0:
        return 7  # Suburban/urban transition
    elif alr < 300.0:
        return 8  # City sky
    else:
        return 9  # Inner-city sky


@pytest.mark.skipif(
    "ALR_PARIS_RADIANCE" not in os.environ,
    reason="Requires ALR_PARIS_RADIANCE env var pointing to a VIIRS GeoTIFF of Paris"
)
def test_paris_bortle_8_9():
    """Paris should be Bortle 8-9 with calibrated C."""
    radiance_path = os.environ["ALR_PARIS_RADIANCE"]
    result = radiance_to_alr(
        radiance_path,
        equal_area_epsg=3035,  # ETRS89-LAEA Europe
        work_resolution_m=450,
    )

    data = result.data
    profile = result.profile
    transform = profile["transform"]

    # Guard: Paris test requires the OUTPUT raster to be in WGS84.
    # The function reprojects back to the input's original CRS, so if the input
    # was a projected CRS the output will be too — fail with a clear message.
    out_crs = profile.get("crs")
    if out_crs is None or out_crs.to_epsg() != 4326:
        pytest.skip(
            f"Paris test requires WGS84 (EPSG:4326) input/output; got {out_crs}. "
            f"Provide a WGS84 VIIRS GeoTIFF (e.g., from EOG VNP46A1 tiles)."
        )

    # Use rasterio's rowcol() to correctly handle any affine transform (rotation,
    # skew, etc.). lon/lat must be in the same CRS as the raster (WGS84 here).
    row, col = rasterio.transform.rowcol(profile["transform"], PARIS_LON, PARIS_LAT)

    assert 0 <= row < data.shape[0], f"Row {row} out of bounds"
    assert 0 <= col < data.shape[1], f"Col {col} out of bounds"

    alr = data[row, col]
    bortle = alr_to_bortle(alr)

    assert 8 <= bortle <= 9, \
        f"Paris ALR={alr:.2f} → Bortle {bortle}, expected 8-9. " \
        f"Recalibrate ALR_CALIB_C per pipeline spec §6."
