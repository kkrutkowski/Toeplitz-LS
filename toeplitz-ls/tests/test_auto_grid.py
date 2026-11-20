from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np

try:
    from mpmath import mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    mp = None

from toeplitz_ls import ihs, ihsdd, ihsf, tls, tlsdd, tlsf
from toeplitz_ls._fastchi2 import _auto_grid, trim_mantissa


class TestAutoGrid(unittest.TestCase):
    def test_explicit_fmin_uses_frequency_units_long_span(self):
        delta_t = 1000.0
        oversampling = 5.0
        nterms = 3
        expected_df = trim_mantissa(1.0 / (oversampling * delta_t * nterms), 5)

        for fmin, fmax in [(0.1, 12.0), (6.0, 12.0), (10.0, 12.0), (11.5, 12.0)]:
            with self.subTest(fmin=fmin, fmax=fmax):
                nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms)
                self.assertEqual(df, expected_df)
                self.assertGreaterEqual(f0, fmin)
                self.assertLess(f0 - df, fmin)
                self.assertEqual(f0, round(f0 / df) * df)
                f_last = f0 + (nf - 1) * df
                self.assertLessEqual(f_last, fmax)
                self.assertGreater(f0 + nf * df, fmax)

    def test_explicit_fmin_uses_frequency_units_short_span(self):
        # With delta_t < 1, dividing fmin by delta_t would push f0 above fmax.
        delta_t = 0.05
        fmin = 0.5
        fmax = 5.0
        nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling=5.0, nterms=3)

        self.assertGreaterEqual(f0, fmin)
        self.assertLess(f0 - df, fmin)
        self.assertGreater(nf, 0)
        self.assertLessEqual(f0 + (nf - 1) * df, fmax)
        self.assertGreater(f0 + nf * df, fmax)

    def test_explicit_fmin_zero_and_exact_multiple(self):
        delta_t = 50.0
        nf_zero, df, f0_zero = _auto_grid(delta_t, 0.0, 2.0, oversampling=5.0, nterms=2)
        self.assertEqual(f0_zero, 0.0)
        self.assertGreater(nf_zero, 0)

        exact_fmin = 64.0 * df
        nf_exact, df_exact, f0_exact = _auto_grid(delta_t, exact_fmin, 2.0, oversampling=5.0, nterms=2)
        self.assertEqual(df_exact, df)
        self.assertEqual(f0_exact, exact_fmin)
        self.assertEqual(nf_exact, nf_zero - 64)

    def test_default_fmin_none_starts_above_rayleigh_frequency(self):
        delta_t = 250.0
        nf, df, f0 = _auto_grid(delta_t, None, 4.0, oversampling=5.0, nterms=3)
        rayleigh = 1.0 / delta_t

        self.assertGreater(f0, rayleigh)
        self.assertLessEqual(f0 - df, rayleigh)
        self.assertEqual(nf, math.floor((4.0 - f0) / df) + 1)

    def test_invalid_fmin_rejected(self):
        with self.assertRaises(ValueError):
            _auto_grid(100.0, -0.1, 2.0, oversampling=5.0, nterms=1)
        with self.assertRaises(ValueError):
            _auto_grid(100.0, 3.0, 2.0, oversampling=5.0, nterms=1)

    def test_autopower_respects_fmin_and_fmax(self):
        rng = np.random.default_rng(seed=123)
        t = np.sort(rng.uniform(0.0, 100.0, 96)).astype(np.float64)
        dy = rng.uniform(0.1, 0.3, t.size).astype(np.float64)
        y = (np.sin(2.0 * np.pi * 1.25 * t) + rng.normal(0.0, dy)).astype(np.float64)

        fmin = 1.0
        fmax = 1.5
        delta_t = float(t.max() - t.min())
        expected_nf, expected_df, expected_f0 = _auto_grid(delta_t, fmin, fmax, oversampling=5.0, nterms=2)

        for mod in (tlsf, tls, ihsf, ihs):
            with self.subTest(module=mod.__name__):
                freq, power = mod.autopower(t, y, dy, fmin=fmin, fmax=fmax, oversampling=5.0, nterms=2)
                self.assertEqual(len(freq), expected_nf)
                self.assertEqual(len(power), expected_nf)
                self.assertAlmostEqual(float(freq[0]), expected_f0, places=15)
                self.assertGreaterEqual(float(freq[0]), fmin)
                self.assertLess(float(freq[0]) - expected_df, fmin)
                self.assertLessEqual(float(freq[-1]), fmax)

        if mp is not None:
            t_small = t[:32]
            y_small = y[:32]
            dy_small = dy[:32]
            delta_t_small = float(t_small.max() - t_small.min())
            dd_nf, dd_df, dd_f0 = _auto_grid(delta_t_small, fmin, fmax, oversampling=5.0, nterms=2)
            for mod in (tlsdd, ihsdd):
                with self.subTest(module=mod.__name__):
                    freq, power = mod.autopower(t_small, y_small, dy_small, fmin=fmin, fmax=fmax, oversampling=5.0, nterms=2)
                    self.assertEqual(len(freq), dd_nf)
                    self.assertEqual(len(power), dd_nf)
                    self.assertAlmostEqual(float(freq[0]), dd_f0, places=15)
                    self.assertGreaterEqual(float(freq[0]), fmin)
                    self.assertLess(float(freq[0]) - dd_df, fmin)
                    self.assertLessEqual(float(freq[-1]), fmax)


if __name__ == "__main__":
    unittest.main()
