import numpy as np
from nightskyquality import radiance_to_alr


def test_point_source_alr_monotonic(tmp_path):
    """Point source at center → ALR decreases radially from center.

    Uses max_km=50 (R_px=111) so valid region fits inside 1000×1000 raster.
    Input IS EPSG:3035 equal-area → no reprojection. WGS84 round-trip is no-op.
    """
    from tests.conftest import point_source_raster
    path = point_source_raster(tmp_path)

    result = radiance_to_alr(
        str(path),
        equal_area_epsg=3035,
        rings=38,
        max_km=50,          # R_px = int(50*1000/450) = 111
        work_resolution_m=450,
    )

    alr = result.data
    H, W = alr.shape
    R_px = int(50 * 1000 / 450)
    center_y, center_x = H // 2, W // 2

    # Valid radius (center to inner NaN edge) — sample in the strong signal region
    # where ALR is well above machine epsilon (r < 80 px from center).
    inner_radius = 80  # ~30 km at 450 m resolution, signal ~ 1e-3 or higher
    radii = np.arange(10, inner_radius, 5)

    directions = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]

    for dy, dx in directions:
        samples = []
        for r in radii:
            if dx == 0 and dy == 0:
                y, x = center_y, center_x
            elif dx == 0:
                y = center_y + r * (1 if dy > 0 else -1)
                x = center_x
            elif dy == 0:
                y = center_y
                x = center_x + r * (1 if dx > 0 else -1)
            else:
                step = r / np.sqrt(2)
                y = center_y + int(step * (1 if dy > 0 else -1))
                x = center_x + int(step * (1 if dx > 0 else -1))
            y = np.clip(y, 0, H - 1)
            x = np.clip(x, 0, W - 1)
            samples.append(alr[y, x])
        samples = np.array(samples)

        # All ALR values > 0 in strong-signal region
        assert np.all(samples > 1e-12), \
            f"Direction ({dy},{dx}): got non-positive ALR, samples={samples[:5]}"

        # Monotonically decreasing (allow small noise within 0.1% of mean)
        diffs = np.diff(samples)
        mean_val = np.mean(samples)
        assert np.all(diffs <= mean_val * 1e-3), \
            f"Direction ({dy},{dx}): ALR not monotonic, diffs={diffs[:5]}"

    # With eps=0.001, the master kernel excludes the center pixel from ring 0,
    # so ALR at the point source location is ~0. Nearby pixels are positive.
    center_val = alr[center_y, center_x]
    near_val = alr[center_y, center_x + 15]  # 15 px away = ~6.75 km
    assert abs(center_val) < 1e-6, f"center ALR should be ~0, got {center_val}"
    assert near_val > 0, f"near-away ALR should be > 0, got {near_val}"
