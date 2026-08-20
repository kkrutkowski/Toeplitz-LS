from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
from mpmath import mp

from toeplitz_ls import tls, tlsdd, tlsf


class TestNormalizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.dps = 30
        rng = np.random.default_rng(seed=42)
        cls.t = np.sort(rng.uniform(0.0, 50.0, 128)).astype(np.float64)
        cls.dy = rng.uniform(0.1, 0.4, cls.t.size).astype(np.float64)
        y_clean = 1.5 * np.sin(2.0 * np.pi * 0.8 * cls.t + 0.3) + 0.8 * np.cos(2.0 * np.pi * 1.6 * cls.t - 0.2)
        cls.y = (y_clean + rng.normal(0.0, cls.dy)).astype(np.float64)

        cls.f0 = 0.1
        cls.df = 0.01
        cls.nf = 64

    def test_d1_gls_condition_and_normalizations(self):
        """Test d=1 (GLS) across all precisions and normalizations."""
        for precision, mod in [("tlsf", tlsf), ("tls", tls)]:
            for norm in ["standard", "asymptotic", "nll", "bayes"]:
                power, cond = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=1,
                    normalization=norm,
                    autonan=False,
                )
                self.assertEqual(len(power), self.nf)
                self.assertEqual(len(cond), self.nf)
                self.assertTrue(np.all(np.isfinite(power)))
                self.assertTrue(np.all(np.isfinite(cond)))
                # Condition for d=1 must be >= 1.0
                self.assertTrue(np.all(cond >= 1.0 - 1e-6))
                if norm == "standard":
                    self.assertTrue(np.all((power >= 0.0) & (power < 1.0)))

    def test_asymptotic_matches_formula(self):
        """Verify asymptotic normalization matches F = (d2 * R2) / (d1 * (1 - R2))."""
        m = self.t.size
        for degree in [1, 2, 3]:
            d1 = 2 * degree + 1
            d2 = m - 1 - d1
            solver = None if degree == 1 else "levinson"

            p_std = tls.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=degree,
                normalization="standard",
                solver=solver,
            )
            p_asymp = tls.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=degree,
                normalization="asymptotic",
                solver=solver,
            )

            expected_asymp = (d2 * p_std) / (d1 * (1.0 - p_std))
            np.testing.assert_allclose(p_asymp, expected_asymp, rtol=1e-5, atol=1e-7)

    def test_nll_and_bayes_d2_d3(self):
        """Verify NLL and Bayes normalizations for multi-harmonic systems (d=2, d=3)."""
        for degree in [2, 3]:
            for solver in ["levinson", "zohar", "bareiss"]:
                for mod in [tlsf, tls]:
                    p_nll, cond = mod.power(
                        self.nf,
                        self.df,
                        self.f0,
                        self.t,
                        self.y,
                        self.dy,
                        nterms=degree,
                        normalization="nll",
                        solver=solver,
                        autonan=False,
                    )
                    self.assertTrue(np.all(np.isfinite(p_nll)))
                    self.assertTrue(np.all(p_nll >= 0.0))
                    self.assertTrue(np.all(cond >= 1.0 - 1e-6))

                    p_bayes = mod.power(
                        self.nf,
                        self.df,
                        self.f0,
                        self.t,
                        self.y,
                        self.dy,
                        nterms=degree,
                        normalization="bayes",
                        solver=solver,
                    )
                    self.assertTrue(np.all(np.isfinite(p_bayes)))
                    self.assertTrue(np.all(p_bayes >= 0.0))

    def test_bayes_disallows_non_toeplitz_solvers(self):
        """Verify that requesting bayes with non-Toeplitz solvers (ldlt, svd) for d > 1 fails."""
        for degree in [2, 3]:
            for solver in ["ldlt", "svd"]:
                with self.assertRaises(RuntimeError) as ctx:
                    tls.power(
                        self.nf,
                        self.df,
                        self.f0,
                        self.t,
                        self.y,
                        self.dy,
                        nterms=degree,
                        normalization="bayes",
                        solver=solver,
                    )
                self.assertIn("invalid solver", str(ctx.exception))

    def test_tlsdd_all_normalizations(self):
        """Verify tlsdd (double-double mpmath) across all normalizations."""
        for norm in ["standard", "asymptotic", "nll", "bayes"]:
            power, cond = tlsdd.power(
                16,
                self.df,
                self.f0,
                self.t[:32],
                self.y[:32],
                self.dy[:32],
                nterms=2,
                normalization=norm,
                solver="levinson",
                autonan=False,
            )
            self.assertEqual(len(power), 16)
            self.assertEqual(len(cond), 16)
            for p, c in zip(power, cond):
                self.assertTrue(mp.isfinite(p))
                self.assertTrue(mp.isfinite(c))
                self.assertTrue(c >= mp.mpf("0.999"))
                self.assertTrue(p >= 0)

    def test_autopower_normalizations(self):
        """Verify autopower helper supports normalization."""
        for norm in ["standard", "asymptotic", "nll", "bayes"]:
            freq, power = tls.autopower(
                self.t,
                self.y,
                self.dy,
                fmin=0.1,
                fmax=2.0,
                nterms=2,
                normalization=norm,
                solver="levinson",
            )
            self.assertTrue(len(freq) > 0)
            self.assertEqual(len(freq), len(power))
            self.assertTrue(np.all(np.isfinite(power)))


if __name__ == "__main__":
    unittest.main()
