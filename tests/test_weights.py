import numpy as np
from nightskyquality._weights import alpha, ring_boundaries_px, ring_midpoints_km, ring_weights


class TestAlpha:
    def test_at_350km_equals_base(self):
        assert alpha(350.0, 2.3, 0.28) == 2.3

    def test_increases_with_distance(self):
        vals = [alpha(d, 2.3, 0.28) for d in [1, 50, 150, 300]]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))


class TestRingBoundaries:
    def test_returns_rings_plus_one_elements(self):
        b = ring_boundaries_px(300, 450, 38)
        assert len(b) == 39
        assert b.dtype.kind == 'i'

    def test_starts_at_zero_ends_at_max_px(self):
        b = ring_boundaries_px(300, 450, 38)
        assert b[0] == 0
        assert b[-1] == int(300 * 1000 / 450)  # 666

    def test_strictly_increasing(self):
        b = ring_boundaries_px(300, 450, 38)
        assert np.all(np.diff(b) > 0)


class TestRingMidpoints:
    def test_length_equals_rings(self):
        b = ring_boundaries_px(300, 450, 38)
        m = ring_midpoints_km(b, 450)
        assert len(m) == 38

    def test_all_positive_and_increasing(self):
        b = ring_boundaries_px(300, 450, 38)
        m = ring_midpoints_km(b, 450)
        assert np.all(m > 0)
        assert np.all(np.diff(m) > 0)


class TestRingWeights:
    def test_count_matches_rings(self):
        b = ring_boundaries_px(300, 450, 38)
        m = ring_midpoints_km(b, 450)
        w = ring_weights(m, 2.3, 0.28)
        assert len(w) == 38

    def test_weights_strictly_decreasing(self):
        b = ring_boundaries_px(300, 450, 38)
        m = ring_midpoints_km(b, 450)
        w = ring_weights(m, 2.3, 0.28)
        assert np.all(np.diff(w) < 0)

    def test_all_positive(self):
        b = ring_boundaries_px(300, 450, 38)
        m = ring_midpoints_km(b, 450)
        w = ring_weights(m, 2.3, 0.28)
        assert np.all(w > 0)
