from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
from mpmath import mp

from toeplitz_ls import ihs, ihsdd, ihsf, tls
from toeplitz_ls._ihs import _check_ihs_normalization


def _wilkie_correct(power, n_eff):
    """Vectorized finite-sample Rayleigh/Wilkie correction from the C kernel."""
    power = np.asarray(power, dtype=np.float64)
    if not (n_eff > 0.0) or not math.isfinite(n_eff):
        return power.copy()
    inv_n = 1.0 / n_eff
    inv_n2 = inv_n * inv_n
    z2 = power * power
    z3 = z2 * power
    z4 = z3 * power
    delta = (2.0 * power - z2) * 0.25 * inv_n
    delta -= (24.0 * power - 132.0 * z2 + 76.0 * z3 - 9.0 * z4) * inv_n2 / 288.0
    valid = np.isfinite(delta) & (delta > -1.0)
    corrected = np.where(valid, power - np.log1p(np.where(valid, delta, 0.0)), power)
    return np.where(np.isfinite(corrected) & (corrected > 0.0), corrected, power)


def _erlang_correct(total, degree):
    result = np.zeros(np.shape(total), dtype=np.float64)
    for idx, value in enumerate(np.ravel(total)):
        if value <= 0.0:
            continue
        log_sum = 1.0
        term = 1.0
        for i in range(1, degree):
            term *= value / i
            log_sum += term
        result.flat[idx] = max(value - math.log(log_sum), 0.0)
    return result


def _ihs_reference(t, y, dy, freqs, nterms, normalization):
    """Direct O(Nf * nterms * M) reference for the IHS statistics."""
    w = 1.0 / (dy * dy)
    ws = w.sum()
    ymean = np.sum(w * y) / ws
    yc = y - ymean
    chi2_ref = np.sum(w * yc * yc)
    wy = w * yc

    wy_abs = np.abs(wy).sum()
    wy_sq = np.sum(wy * wy)
    norm_neff = wy_abs * wy_abs / wy_sq
    nll_scale = norm_neff / (wy_abs * wy_abs)

    freqs = np.asarray(freqs, dtype=np.float64)
    accumulated = np.zeros(freqs.size, dtype=np.float64)
    corrected = np.zeros(freqs.size, dtype=np.float64)
    for k in range(1, nterms + 1):
        z = np.exp(-2j * np.pi * np.outer(k * freqs, t)) @ wy
        raw = np.abs(z) ** 2
        accumulated += raw
        corrected += _wilkie_correct(raw * nll_scale, norm_neff)

    if normalization in ("raw", "standard"):
        return 2.0 * accumulated / (ws * chi2_ref)
    if normalization in ("corrected", "nll"):
        return _erlang_correct(corrected, nterms)
    if normalization == "default":
        return _erlang_correct(accumulated * nll_scale, nterms)
    raise ValueError("unknown normalization: %s" % normalization)


class TestIHS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mp.dps = 30
        rng = np.random.default_rng(seed=7)
        cls.t = np.sort(rng.uniform(0.0, 50.0, 96)).astype(np.float64)
        cls.dy = rng.uniform(0.1, 0.4, cls.t.size).astype(np.float64)
        y_clean = 1.2 * np.sin(2.0 * np.pi * 0.8 * cls.t + 0.3) + 0.7 * np.cos(2.0 * np.pi * 1.6 * cls.t - 0.2)
        cls.y = (y_clean + rng.normal(0.0, cls.dy)).astype(np.float64)
        cls.f0 = 0.1
        cls.df = 0.01
        cls.nf = 24

    def _freqs(self):
        return self.f0 + self.df * np.arange(self.nf, dtype=np.float64)

    def test_raw_matches_direct_reference(self):
        for precision, mod, rtol in [
            ("ihsf", ihsf, 1e-4),
            ("ihs", ihs, 1e-9),
        ]:
            for degree in [1, 2, 3]:
                power = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=degree,
                    normalization="raw",
                )
                expected = _ihs_reference(self.t, self.y, self.dy, self._freqs(), degree, "raw")
                np.testing.assert_allclose(
                    power,
                    expected,
                    rtol=rtol,
                    atol=1e-7,
                    err_msg=f"{precision} raw d={degree} mismatch",
                )

    def test_raw_matches_gls_for_orthogonal_single_harmonic(self):
        """At an orthogonal Fourier frequency IHS raw reduces to GLS R^2."""
        m = 128
        t = np.arange(m, dtype=np.float64)
        frequency = 17.0 / m
        y = np.sin(2.0 * np.pi * frequency * t) + 0.5 * np.cos(2.0 * np.pi * frequency * t)
        ihs_power = ihs.power(8, 1.0 / m, frequency, t, y, nterms=1, normalization="raw")
        tls_power = tls.power(8, 1.0 / m, frequency, t, y, nterms=1, normalization="standard")
        np.testing.assert_allclose(ihs_power[0], tls_power[0], rtol=1e-9, atol=1e-12)

    def test_corrected_and_raw_can_choose_different_peaks(self):
        """Per-harmonic Wilkie correction makes NLL non-monotonic in the sum."""
        rng = np.random.default_rng(seed=3)
        t = np.sort(rng.uniform(0.0, 50.0, 128))
        dy = np.ones(t.size)
        y = (
            3.0 * np.sin(2.0 * np.pi * 0.8 * t + 0.3)
            + 2.0 * np.sin(2.0 * np.pi * 1.6 * t + 1.1)
            + 1.5 * np.sin(2.0 * np.pi * 2.4 * t + 2.0)
            + rng.normal(0.0, 0.2, t.size)
        )
        raw = ihs.power(500, 0.002, 0.1, t, y, dy, nterms=3, normalization="raw")
        corrected = ihs.power(500, 0.002, 0.1, t, y, dy, nterms=3, normalization="corrected")
        self.assertNotEqual(int(np.argmax(raw)), int(np.argmax(corrected)))
        self.assertAlmostEqual(0.1 + 0.002 * int(np.argmax(corrected)), 0.8, places=12)

    def test_corrected_matches_direct_reference(self):
        for precision, mod, rtol in [
            ("ihsf", ihsf, 2e-4),
            ("ihs", ihs, 1e-9),
        ]:
            for degree in [1, 2, 3]:
                power = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=degree,
                    normalization="corrected",
                )
                expected = _ihs_reference(self.t, self.y, self.dy, self._freqs(), degree, "corrected")
                np.testing.assert_allclose(
                    power,
                    expected,
                    rtol=rtol,
                    atol=1e-7,
                    err_msg=f"{precision} corrected d={degree} mismatch",
                )

    def test_default_omits_the_wilkie_correction(self):
        """Default is the initial formula + Erlang, without Wilkie."""
        default = ihs.power(self.nf, self.df, self.f0, self.t, self.y, self.dy, nterms=2, normalization="default")
        corrected = ihs.power(self.nf, self.df, self.f0, self.t, self.y, self.dy, nterms=2, normalization="corrected")
        self.assertTrue(np.all(np.isfinite(default)))
        self.assertFalse(np.allclose(default, corrected))

    def test_raw_sums_single_harmonic_contributions(self):
        degree = 3
        total = ihs.power(
            self.nf,
            self.df,
            self.f0,
            self.t,
            self.y,
            self.dy,
            nterms=degree,
            normalization="raw",
        )
        # Each nterms=k call returns the sum through harmonic k, so differences
        # recover the individual harmonic contributions.
        summed = ihs.power(
            self.nf,
            self.df,
            self.f0,
            self.t,
            self.y,
            self.dy,
            nterms=1,
            normalization="raw",
        )
        for harmonic in range(2, degree + 1):
            current = ihs.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=harmonic,
                normalization="raw",
            )
            previous = ihs.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=harmonic - 1,
                normalization="raw",
            )
            summed += current - previous
        np.testing.assert_allclose(total, summed, rtol=1e-10, atol=1e-12)

    def test_default_uncertainties_and_multiblock_grid(self):
        """Exercise dy=None and a grid spanning several NuFFT blocks."""
        nf = 300
        freqs = self.f0 + self.df * np.arange(nf, dtype=np.float64)
        dy = np.ones_like(self.t)
        for precision, mod, rtol in [
            ("ihsf", ihsf, 1e-4),
            ("ihs", ihs, 1e-9),
        ]:
            for normalization in ["raw", "corrected", "default"]:
                power = mod.power(
                    nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    nterms=2,
                    normalization=normalization,
                )
                expected = _ihs_reference(self.t, self.y, dy, freqs, 2, normalization)
                if normalization == "raw":
                    # nterms > 1 is a sum of per-harmonic contributions, not a
                    # bounded R^2; at least one bin here is legitimately > 1.
                    self.assertGreater(float(np.max(expected)), 1.0)
                np.testing.assert_allclose(
                    power,
                    expected,
                    rtol=rtol,
                    atol=1e-7,
                    err_msg=f"{precision} {normalization} multiblock mismatch",
                )

    def test_warns_when_nterms4_exceeds_neff(self):
        """IHS is not suitable when d^4 > Neff; the wrapper must warn."""
        with self.assertWarns(RuntimeWarning):
            ihs.power(
                16,
                self.df,
                self.f0,
                self.t[:16],
                self.y[:16],
                self.dy[:16],
                nterms=3,
                normalization="default",
            )

    def test_backends_agree(self):
        for normalization in ["raw", "corrected", "default"]:
            reference = None
            for backend in ["pswf43", "pswf21", "lra"]:
                power = ihs.power(
                    32,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    backend=backend,
                    nterms=2,
                    normalization=normalization,
                )
                self.assertTrue(np.all(np.isfinite(power)))
                self.assertTrue(np.all(power >= 0.0))
                if reference is None:
                    reference = power
                else:
                    np.testing.assert_allclose(
                        power,
                        reference,
                        rtol=2e-3,
                        atol=1e-7,
                        err_msg=f"backend {backend} {normalization} mismatch",
                    )

    def test_dd_matches_double(self):
        for normalization in ["raw", "corrected", "default"]:
            power_dd = ihsdd.power(
                16,
                self.df,
                self.f0,
                self.t[:48],
                self.y[:48],
                self.dy[:48],
                nterms=2,
                normalization=normalization,
            )
            power_d = ihs.power(
                16,
                self.df,
                self.f0,
                self.t[:48],
                self.y[:48],
                self.dy[:48],
                nterms=2,
                normalization=normalization,
            )
            self.assertEqual(len(power_dd), 16)
            for value in power_dd:
                self.assertTrue(mp.isfinite(value))
                self.assertTrue(value >= 0)

            np.testing.assert_allclose(
                [float(value) for value in power_dd],
                power_d,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"ihsdd {normalization} mismatch",
            )

    def test_autopower(self):
        for mod in [ihsf, ihs]:
            for normalization in ["raw", "corrected", "default"]:
                freq, power = mod.autopower(
                    self.t,
                    self.y,
                    self.dy,
                    fmin=0.1,
                    fmax=2.0,
                    nterms=2,
                    normalization=normalization,
                )
                self.assertGreater(len(freq), 0)
                self.assertEqual(len(freq), len(power))
                self.assertTrue(np.all(np.isfinite(power)))
                self.assertTrue(np.all(power >= 0.0))

        freq, power = ihsdd.autopower(
            self.t[:32],
            self.y[:32],
            self.dy[:32],
            fmin=0.1,
            fmax=1.0,
            nterms=2,
            normalization="corrected",
        )
        self.assertEqual(len(freq), len(power))
        self.assertTrue(all(mp.isfinite(value) for value in power))

    def test_invalid_normalizations(self):
        for mod in [ihsf, ihs]:
            for normalization in ["anova", "regularized", "bayes", "asymptotic", "bogus"]:
                with self.assertRaises(ValueError) as ctx:
                    mod.power(
                        self.nf,
                        self.df,
                        self.f0,
                        self.t,
                        self.y,
                        self.dy,
                        nterms=2,
                        normalization=normalization,
                    )
                message = str(ctx.exception)
                self.assertIn("raw", message)
                self.assertIn("corrected", message)
                self.assertIn("default", message)
                self.assertNotIn("anova", message)
                self.assertNotIn("regularized", message)

        self.assertEqual(_check_ihs_normalization("raw"), 0)
        self.assertEqual(_check_ihs_normalization("corrected"), 2)
        self.assertEqual(_check_ihs_normalization("default"), 4)
        self.assertEqual(_check_ihs_normalization("standard"), 0)
        self.assertEqual(_check_ihs_normalization("nll"), 2)
        with self.assertRaises(ValueError):
            _check_ihs_normalization(1)
        with self.assertRaises(ValueError):
            _check_ihs_normalization("bogus")


if __name__ == "__main__":
    unittest.main()
