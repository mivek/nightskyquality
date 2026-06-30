# nightskyquality

Modernized Python library for computing the **All-sky Light pollution Ratio (ALR)** from VIIRS Day/Night Band radiance GeoTIFFs.

**ALR** is a unitless ratio of anthropogenic to natural sky brightness over the entire hemisphere of vision, as defined by [Duriscoe et al. (2018)](https://www.researchgate.net/publication/324789721_A_simplified_model_of_all-sky_artificial_sky_glow_derived_from_VIIRS_DayNight_band_data).

This library implements the master-kernel FFT convolution approach: precompute one weighted kernel `K = C · Σ wᵢ · kᵢ`, then a single FFT convolution per tile. This is ~38× faster than per-ring convolution and produces identical numerical results.

---

## Credits

- **Katy Abbott** — original implementation (Geoscientists-in-the-Parks intern at Big Bend National Park, 2019–2020). See the original [BigBendNP/nightskyquality](https://github.com/BigBendNP/nightskyquality) repository (MIT).
- **Original fork source:** [katyabbott/nps-night-rad](https://github.com/katyabbott/nps-night-rad)
- **Scientific basis:** Duriscoe, D. et al. (2018). *A simplified model of all-sky artificial sky glow derived from VIIRS Day/Night band data.* Journal of Quantitative Spectroscopy and Radiative Transfer, 214, 133–145.
- **Calibration method:** Upstream pipeline spec §6 describes calibration via control points (dark-sky reference + heavily light-polluted reference).

---

## Installation

```bash
# Development install (from repo root)
pip install -e .

# Once published to PyPI:
pip install nightskyquality
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Rasterio ≥ 1.3, Shapely ≥ 2.0.

> **Note on Shapely:** Shapely is included as a dependency for future ROI clipping support per the upstream design's optional geopandas/shapely buffer path. The current implementation does NOT use Shapely — ALR is calculated over the full raster extent with no clipping.

---

## Quick start

```python
from nightskyquality import radiance_to_alr

result = radiance_to_alr(
    radiance_path="path/to/viirs_monthly_composite.tif",
    equal_area_epsg=3035,       # ETRS89-LAEA for Europe
    work_resolution_m=450,
    rings=38,
    max_km=300,
    alpha_base=2.3,
    alpha_exp=0.28,
    calib_c=1/562.72,
    noise_floor=0.5,
)

# result.data   — 2D numpy array (float64) of ALR values
# result.profile — rasterio profile dict for writing to GeoTIFF

# Write result to disk:
import rasterio
with rasterio.open("alr_output.tif", "w", **result.profile) as dst:
    dst.write(result.data, 1)
```

The function reads the input GeoTIFF, reprojects to the specified equal-area CRS at the working resolution, computes ALR via FFT convolution, then reprojects back to the **input's original CRS**. No files are written — the result is returned in memory.

---

## ALR formula

```
ALR(p) = C · Σᵢ w(rᵢ) · Σ_{q ∈ annulus i} radiance(q)

where:
  w(r) = r^(-α(r))
  α(r) = α_base · (r / 350)^α_exp
  C    = calibration constant (default 1/562.72, from Duriscoe US calibration)
```

The master kernel `K` pre-computes the weighted sum across all rings, enabling a single FFT convolution per tile.

### EPSG guidance

| Region | Recommended equal-area projection | EPSG |
|--------|-----------------------------------|------|
| Europe | ETRS89-LAEA | 3035 |
| North America | NAD83 / Conus Albers | 5070 |
| Alaska | NAD83 / Alaska Albers | 3338 |
| Global (land) | WGS 84 / World Mollweide | 54009 |
| Arctic | NSIDC Sea Ice Polar Stereographic North | 3413 |
| Antarctica | NSIDC Sea Ice Polar Stereographic South | 3976 |

---

## Europe calibration

The default calibration constant `C = 1/562.72` is derived from Duriscoe's observational data in the western United States. For **relative ranking** of ALR within a region, this constant is sufficient.

For **absolute Bortle classification** in European conditions, the constant may need recalibration via control points (see upstream pipeline spec §6):

1. Select a dark-sky reference site (known Bortle 1-2) and a heavily light-polluted site (Bortle 8-9).
2. Run `radiance_to_alr` with `calib_c=1.0` to get uncalibrated raw Σ values.
3. Solve for `C` such that the raw values map to the expected ALR thresholds.

The included geographic validation test (`tests/test_geographic_validation.py`) uses Paris (Bortle 8-9) as a reference — it is skipped by default and requires a real VIIRS radiance GeoTIFF.

---

## Tiling

Large rasters are processed in tiles of 2000×2000 pixels with `R_px` overlap (where `R_px = max_km * 1000 / work_resolution_m`). Each tile is convolved independently, then the valid (non-overlapping) regions are assembled into the final mosaic. This keeps memory usage bounded regardless of input size.

---

## Testing

```bash
pytest tests/ -v
```

Run the geographic validation test (requires real VIIRS data):

```bash
ALR_PARIS_RADIANCE=/path/to/paris_viirs.tif pytest tests/test_geographic_validation.py -v
```

---

## License

MIT License — Copyright (c) 2020 Katy Abbott. See [LICENSE.txt](./LICENSE.txt).
