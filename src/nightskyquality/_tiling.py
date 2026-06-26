import numpy as np
from ._types import TileSpec


def tile_grid(height: int, width: int, tile_size: int, r_px: int) -> list[TileSpec]:
    """Partition raster into overlapping tiles. Each tile has a read_window
    (tile_size + 2*r_px) and a valid_slice (inner tile_size portion).
    Right/bottom tiles may be smaller (partial)."""
    specs = []
    for row_start in range(0, height, tile_size):
        for col_start in range(0, width, tile_size):
            r0 = max(0, row_start - r_px)
            r1 = min(height, row_start + tile_size + r_px)
            c0 = max(0, col_start - r_px)
            c1 = min(width, col_start + tile_size + r_px)
            vr_start = row_start - r0
            vr_end = vr_start + min(tile_size, height - row_start)
            vc_start = col_start - c0
            vc_end = vc_start + min(tile_size, width - col_start)
            specs.append(TileSpec(
                read_window=(r0, r1, c0, c1),
                valid_slice=(slice(vr_start, vr_end), slice(vc_start, vc_end))
            ))
    return specs


def assemble_mosaic(
    tiles: list[np.ndarray],
    tile_specs: list[TileSpec],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Stitch valid regions into full output raster."""
    result = np.full(full_shape, np.nan, dtype=np.float64)
    for tile, spec in zip(tiles, tile_specs):
        r0, r1, c0, c1 = spec.read_window
        vr, vc = spec.valid_slice
        out_r0 = r0 + vr.start
        out_r1 = r0 + vr.stop
        out_c0 = c0 + vc.start
        out_c1 = c0 + vc.stop
        result[out_r0:out_r1, out_c0:out_c1] = tile[vr, vc]
    return result
