import unittest

import numpy as np

from methods import METHODS, UnsupportedMethodError, make_frequency_grid


class ComparisonMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(913)
        cls.t = np.sort(rng.uniform(0.0, 40.0, 220)).astype(np.float64)
        cls.dy = rng.uniform(0.08, 0.22, cls.t.size).astype(np.float64)
        cls.y = (
            1.8 * np.sin(2.0 * np.pi * 0.217 * cls.t + 0.2)
            + 0.4 * np.cos(2.0 * np.pi * 0.431 * cls.t)
            + rng.normal(0.0, cls.dy)
        ).astype(np.float64)

    def power(self, name, nterms, dy=None):
        grid = make_frequency_grid(self.t, fmax=0.65, oversampling=5, nterms=nterms)
        result = METHODS[name].power(
            grid.Nf,
            grid.df,
            grid.f0,
            self.t,
            self.y,
            dy,
            nterms=nterms,
            frequency=grid.frequency,
        )
        return grid, np.asarray(result)

    def test_aovdist_matches_direct_chi2_unweighted_and_weighted(self):
        for dy in (None, self.dy):
            with self.subTest(weighted=dy is not None):
                _grid, expected = self.power("astropy-slow", 3, dy)
                _grid, actual = self.power("aovdist", 3, dy)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_common_methods_recover_direct_peak(self):
        _grid, reference = self.power("astropy-slow", 3)
        expected_bin = int(np.nanargmax(reference))
        for name in ("tls", "tlsf", "astropy-lra", "astropy-fasper", "aovdist", "fastchi2_103"):
            with self.subTest(method=name):
                _grid, candidate = self.power(name, 3)
                self.assertEqual(int(np.nanargmax(candidate)), expected_bin)

    def test_fastchi2_rejects_non_fft_grid_and_documented_high_order(self):
        grid = make_frequency_grid(self.t, fmax=0.65, oversampling=5, nterms=1)
        with self.assertRaises(ValueError):
            METHODS["fastchi2_103"].power(
                grid.Nf,
                grid.df,
                grid.f0 + 0.25 * grid.df,
                self.t,
                self.y,
                nterms=1,
            )
        with self.assertRaises(UnsupportedMethodError):
            METHODS["fastchi2_103"].power(
                grid.Nf, grid.df, grid.f0, self.t, self.y, nterms=16
            )


if __name__ == "__main__":
    unittest.main()
