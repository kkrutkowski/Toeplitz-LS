from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
from mpmath import mp

from toeplitz_ls import tls, tlsdd, tlsf
from toeplitz_ls._fastchi2 import _check_normalization

PRIMARY_NORMALIZATIONS = ["standard", "anova", "nll", "regularized"]


def _pochhammer_nll(r2, degree, num_observations, r2_max=1.0 - 1e-12):
    """Reference implementation of the C Pochhammer NLL kernel."""
    r2 = np.clip(np.asarray(r2, dtype=np.float64), 0.0, r2_max)
    b = 0.5 * (num_observations - (2 * degree + 1))
    term0 = -b * np.log1p(-r2)
    if degree <= 1:
        return np.maximum(term0, 0.0)

    c = 1.0
    s_minus_1 = 0.0
    for k in range(1, degree):
        c *= (b + (k - 1)) / k * r2
        s_minus_1 += c
    nll = term0 - np.log1p(s_minus_1)
    return np.maximum(nll, 0.0)


def _beta_nll(r2, degree, num_observations):
    """Exact natural-log NLL via the regularized incomplete beta function."""
    b = mp.mpf(num_observations - (2 * degree + 1)) / 2
    r2 = mp.mpf(min(max(float(r2), 1.0e-15), 1.0 - 1.0e-15))
    p_value = mp.betainc(b, mp.mpf(degree), mp.mpf(0), mp.mpf(1) - r2, regularized=True)
    return -mp.log(p_value)


def _cybenko_nll(kappa, degree, num_observations):
    """Python port of the ``d <= 2`` branch of ``nll_cybenko_scalar_d``."""
    d = float(degree)
    m_count = float(num_observations)
    inv_m = 1.0 / m_count
    inv_m2 = inv_m * inv_m

    var_s = 2.0 * d + (8.0 * d * d - 14.0 * d) * inv_m
    var_s += ((184.0 / 3.0) * d * d * d - 98.0 * d * d + (158.0 / 3.0) * d) * inv_m2
    mean_s = 2.0 * d
    k_eff = (mean_s * mean_s) / var_s
    theta = var_s / mean_s

    n_shape = 2.0 * d
    sum_n_eff = n_shape * m_count - n_shape * (n_shape + 1.0)
    sum_n_eff += (n_shape * (n_shape + 1.0) * (2.0 * n_shape + 1.0)) / (6.0 * m_count)

    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa <= 1.0:
        return 0.0
    kappa = min(kappa, 5000.0)
    pow_k = kappa ** (1.0 / (2.0 * d))
    k_val = (pow_k - 1.0) / (pow_k + 1.0)
    s_stat = sum_n_eff * k_val * k_val
    if s_stat <= 0.0:
        return 0.0

    z = s_stat / theta
    order = 2 * degree
    delta = k_eff - float(order)
    c = 1.0
    poly = 1.0
    d_poly = 0.0
    for k in range(1, order):
        c *= z / k
        poly += c
        if abs(delta) > 1.0e-5:
            h_k = sum(1.0 / i for i in range(1, k + 1))
            d_poly -= c * h_k

    nll = z - math.log(poly)
    if abs(delta) > 1.0e-5:
        nll -= delta * (math.log(z) + d_poly / poly)
    return max(nll, 0.0)


def _laplace_combine(delta):
    """Reference implementation of the Laplace/Bayes composition."""
    if delta >= 0.0:
        return delta + math.log(2.0)
    inner = 1.0 - 0.5 * math.exp(delta)
    if inner <= 0.0:
        return 0.0
    return -math.log(inner)


def _reference_regularized(r2, condition_bound, degree, num_observations, r2_max):
    product = condition_bound / (2 * degree + 1)
    nll_data = float(_pochhammer_nll(r2, degree, num_observations, r2_max=r2_max))
    nll_cond = _cybenko_nll(product, degree, num_observations)
    return _laplace_combine(nll_data - nll_cond)


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
            for norm in PRIMARY_NORMALIZATIONS:
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
                # The condition value is n * P with n = 3, so it is >= 3.
                self.assertTrue(np.all(cond >= 3.0 - 1e-6))
                if norm == "standard":
                    self.assertTrue(np.all((power >= 0.0) & (power < 1.0)))

    def test_d1_fast_path_matches_condition_path(self):
        """The fast path must not change power when cond is not requested."""
        # -ffast-math may reassociate the simpler fast loop, so allow a few ULP;
        # the analytic reference tests below are the hard correctness gate.
        for mod, rtol, atol in ((tlsf, 1e-6, 1e-9), (tls, 1e-12, 1e-15)):
            for norm in PRIMARY_NORMALIZATIONS:
                p_fast = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=1,
                    normalization=norm,
                    autonan=True,
                )
                p_full, cond = mod.power(
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
                np.testing.assert_allclose(p_fast, p_full, rtol=rtol, atol=atol)
                self.assertTrue(np.all(np.isfinite(cond)))
                self.assertTrue(np.all(cond >= 3.0 - 1e-6))

    def test_anova_matches_formula(self):
        """Verify ANOVA normalization matches F = (d2 * R2) / (d1 * (1 - R2))."""
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
            p_anova = tls.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=degree,
                normalization="anova",
                solver=solver,
            )

            expected_anova = (d2 * p_std) / (d1 * (1.0 - p_std))
            np.testing.assert_allclose(p_anova, expected_anova, rtol=1e-5, atol=1e-7)

    def test_nll_and_regularized_d2_d3(self):
        """Verify NLL and regularized normalizations for multi-harmonic systems (d=2, d=3)."""
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
                    self.assertTrue(np.all(cond >= 2 * degree + 1 - 1e-6))

                    p_regularized = mod.power(
                        self.nf,
                        self.df,
                        self.f0,
                        self.t,
                        self.y,
                        self.dy,
                        nterms=degree,
                        normalization="regularized",
                        solver=solver,
                    )
                    self.assertTrue(np.all(np.isfinite(p_regularized)))
                    self.assertTrue(np.all(p_regularized >= 0.0))

    def test_regularized_disallows_non_toeplitz_solvers(self):
        """Verify regularized rejects non-Toeplitz solvers (ldlt, svd) for d > 1."""
        for norm in ["regularized", "bayes"]:
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
                            normalization=norm,
                            solver=solver,
                        )
                    self.assertIn("invalid solver", str(ctx.exception))

    def test_tlsdd_all_normalizations(self):
        """Verify tlsdd (double-double mpmath) across all normalizations."""
        for norm in PRIMARY_NORMALIZATIONS:
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
                self.assertTrue(c >= mp.mpf("4.999"))
                self.assertTrue(p >= 0)

    def test_autopower_normalizations(self):
        """Verify autopower helper supports normalization."""
        for norm in PRIMARY_NORMALIZATIONS:
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

    def test_legacy_aliases(self):
        """Verify anova/asymptotic and regularized/bayes are exact aliases."""
        self.assertEqual(_check_normalization("standard"), 0)
        self.assertEqual(_check_normalization("anova"), 1)
        self.assertEqual(_check_normalization("asymptotic"), 1)
        self.assertEqual(_check_normalization("nll"), 2)
        self.assertEqual(_check_normalization("regularized"), 3)
        self.assertEqual(_check_normalization("bayes"), 3)
        self.assertEqual(_check_normalization("Regularized "), 3)

        for mod in [tlsf, tls]:
            p_anova = mod.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=2,
                normalization="anova",
                solver="levinson",
            )
            p_asymptotic = mod.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=2,
                normalization="asymptotic",
                solver="levinson",
            )
            np.testing.assert_array_equal(p_anova, p_asymptotic)

            p_reg, cond_reg = mod.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=2,
                normalization="regularized",
                solver="levinson",
                autonan=False,
            )
            p_bayes, cond_bayes = mod.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=2,
                normalization="bayes",
                solver="levinson",
                autonan=False,
            )
            np.testing.assert_array_equal(p_reg, p_bayes)
            np.testing.assert_array_equal(cond_reg, cond_bayes)

        p_anova = tlsdd.power(
            16,
            self.df,
            self.f0,
            self.t[:32],
            self.y[:32],
            self.dy[:32],
            nterms=2,
            normalization="anova",
            solver="levinson",
        )
        p_asymptotic = tlsdd.power(
            16,
            self.df,
            self.f0,
            self.t[:32],
            self.y[:32],
            self.dy[:32],
            nterms=2,
            normalization="asymptotic",
            solver="levinson",
        )
        self.assertEqual(p_anova, p_asymptotic)

        p_reg, cond_reg = tlsdd.power(
            16,
            self.df,
            self.f0,
            self.t[:32],
            self.y[:32],
            self.dy[:32],
            nterms=2,
            normalization="regularized",
            solver="levinson",
            autonan=False,
        )
        p_bayes, cond_bayes = tlsdd.power(
            16,
            self.df,
            self.f0,
            self.t[:32],
            self.y[:32],
            self.dy[:32],
            nterms=2,
            normalization="bayes",
            solver="levinson",
            autonan=False,
        )
        self.assertEqual(p_reg, p_bayes)
        self.assertEqual(cond_reg, cond_bayes)

    def test_d1_condition_matches_astropy_sums(self):
        """Verify the d=1 condition bound uses both 3x3 Toeplitz reflections."""
        freqs = self.f0 + self.df * np.arange(self.nf)
        weights = 1.0 / (self.dy * self.dy)
        weight_sum = weights.sum()

        for precision, mod, limit, rtol in [
            ("tls", tls, 0.999999999999999, 1e-9),
            ("tlsf", tlsf, 0.9999999, 1e-2),
        ]:
            _, cond = mod.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=1,
                normalization="standard",
                autonan=False,
            )

            expected = []
            for freq in freqs:
                arg = 2.0 * np.pi * freq * self.t
                c_sum = np.sum(weights * np.cos(arg))
                s_sum = np.sum(weights * np.sin(arg))
                c2_sum = np.sum(weights * np.cos(2.0 * arg))
                s2_sum = np.sum(weights * np.sin(2.0 * arg))

                c = c_sum / weight_sum
                s = s_sum / weight_sum
                c2 = c2_sum / weight_sum
                s2 = s2_sum / weight_sum

                k1 = math.hypot(c, s)
                k1 = min(max(k1, 0.0), limit)
                den = 1.0 - c * c - s * s
                if den > 0.0:
                    k2 = math.hypot(c2 - c * c + s * s, s2 - 2.0 * c * s) / den
                    k2 = min(max(k2, 0.0), limit)
                else:
                    k2 = limit
                expected.append(3.0 * (1.0 + k1) / (1.0 - k1) * (1.0 + k2) / (1.0 - k2))

            np.testing.assert_allclose(
                cond,
                np.asarray(expected),
                rtol=rtol,
                atol=1e-8,
                err_msg=f"{precision} d=1 condition bound mismatch",
            )

    def test_regularized_uses_cybenko_product(self):
        """Verify the regularizer consumes P = cond / (2 * degree + 1), not cond."""
        for degree in [1, 2]:
            solver = None if degree == 1 else "levinson"
            for precision, mod, rtol, atol, r2_max in [
                ("tls", tls, 1e-8, 1e-8, 0.9999999999999),
                ("tlsf", tlsf, 5e-3, 1e-3, 0.9999999),
            ]:
                p_std, cond = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=degree,
                    normalization="standard",
                    solver=solver,
                    autonan=False,
                )
                p_reg, _ = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=degree,
                    normalization="regularized",
                    solver=solver,
                    autonan=False,
                )

                expected = np.asarray(
                    [
                        _reference_regularized(
                            float(p_std[i]),
                            float(cond[i]),
                            degree,
                            self.t.size,
                            r2_max=r2_max,
                        )
                        for i in range(self.nf)
                    ]
                )
                np.testing.assert_allclose(
                    p_reg,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{precision} regularized d={degree} product scaling mismatch",
                )

    def test_nll_matches_incomplete_beta(self):
        """Verify NLL matches the exact natural-log FAP from the incomplete beta."""
        for degree in [1, 2, 3]:
            solver = None if degree == 1 else "levinson"
            for precision, mod, rtol, atol in [
                ("tls", tls, 1e-9, 1e-12),
                ("tlsf", tlsf, 1e-3, 1e-5),
            ]:
                p_std = mod.power(
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
                p_nll = mod.power(
                    self.nf,
                    self.df,
                    self.f0,
                    self.t,
                    self.y,
                    self.dy,
                    nterms=degree,
                    normalization="nll",
                    solver=solver,
                )
                expected = np.asarray(
                    [_beta_nll(float(r2), degree, self.t.size) for r2 in p_std],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    p_nll,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"{precision} nll d={degree} beta mismatch",
                )

    def test_invalid_normalization_message(self):
        """Verify only primary normalization names are advertised."""
        with self.assertRaises(ValueError) as ctx:
            tls.power(
                self.nf,
                self.df,
                self.f0,
                self.t,
                self.y,
                self.dy,
                nterms=2,
                normalization="bogus",
                solver="levinson",
            )
        message = str(ctx.exception)
        self.assertIn("regularized", message)
        self.assertIn("anova", message)
        self.assertNotIn("bayes", message)
        self.assertNotIn("asymptotic", message)

        with self.assertRaises(ValueError):
            _check_normalization(4)
        with self.assertRaises(ValueError):
            _check_normalization("bogus")


if __name__ == "__main__":
    unittest.main()
