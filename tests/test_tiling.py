import numpy as np
from nightskyquality._tiling import tile_grid, assemble_mosaic


def test_tile_grid_covers_full_raster():
    specs = tile_grid(height=100, width=100, tile_size=40, r_px=5)
    covered = np.zeros((100, 100), dtype=bool)
    for spec in specs:
        r0, r1, c0, c1 = spec.read_window
        vr, vc = spec.valid_slice
        covered[r0+vr.start:r0+vr.stop, c0+vc.start:c0+vc.stop] = True
    assert np.all(covered)


def test_assemble_mosaic_roundtrip():
    specs = tile_grid(height=10, width=10, tile_size=4, r_px=2)
    tiles = []
    for i, spec in enumerate(specs):
        r0, r1, c0, c1 = spec.read_window
        tiles.append(np.ones((r1 - r0, c1 - c0), dtype=np.float64) * i)
    result = assemble_mosaic(tiles, specs, (10, 10))
    assert result.shape == (10, 10)
    assert not np.any(np.isnan(result))
