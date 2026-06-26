import numpy as np
from nightskyquality._kernel import annulus_kernel, master_kernel
from nightskyquality._config import ALRConfig


class TestAnnulusKernel:
    def test_shape(self):
        k = annulus_kernel(0.001, 10)
        assert k.shape == (21, 21)

    def test_center_pixel_zero(self):
        k = annulus_kernel(0.001, 5)
        center = k.shape[0] // 2
        assert k[center, center] == 0

    def test_values_are_zero_or_one(self):
        k = annulus_kernel(5.0, 10)
        assert np.all((k == 0) | (k == 1))

    def test_radial_symmetry(self):
        k = annulus_kernel(3.0, 8)
        assert np.allclose(k, np.rot90(k, 2))

    def test_ones_form_a_ring(self):
        k = annulus_kernel(5.0, 10)
        n_ones = np.sum(k)
        expected = np.pi * (10**2 - 5**2)
        assert abs(n_ones - expected) < expected * 0.3


class TestMasterKernel:
    def test_shape_matches_2Rpx_plus_1(self):
        cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)
        K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                          cfg.alpha_base, cfg.alpha_exp, cfg.calib_c)
        expected_rpx = int(5 * 1000 / 450)
        assert K.shape == (2 * expected_rpx + 1, 2 * expected_rpx + 1)

    def test_center_is_zero(self):
        cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)
        K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                          cfg.alpha_base, cfg.alpha_exp, cfg.calib_c)
        c = K.shape[0] // 2
        assert K[c, c] == 0.0

    def test_all_positive_or_zero(self):
        cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)
        K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                          cfg.alpha_base, cfg.alpha_exp, cfg.calib_c)
        assert np.all(K >= 0)

    def test_radial_symmetry(self):
        cfg = ALRConfig(rings=4, max_km=5, work_resolution_m=450)
        K = master_kernel(cfg.rings, cfg.max_km, cfg.work_resolution_m,
                          cfg.alpha_base, cfg.alpha_exp, cfg.calib_c)
        assert np.allclose(K, np.rot90(K, 2))
