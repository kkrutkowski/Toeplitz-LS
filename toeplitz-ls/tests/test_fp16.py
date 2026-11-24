"""Tests for IEEE 754 binary16 (FP16) periodogram wrappers (tlsh and ihsh)."""

from __future__ import annotations

import numpy as np
import pytest

from toeplitz_ls import get_peaks, ihs, ihsh, tls, tlsh
from toeplitz_ls._fastchi2 import _load_library


def _make_signal(M=300, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 40.0, size=M)).astype(np.float64)
    f_true = 0.37
    y = 2.0 * np.sin(2.0 * np.pi * f_true * t) + 0.7 * np.cos(2.0 * np.pi * 2.0 * f_true * t + 0.4) + 0.15 * rng.normal(size=M)
    dy = rng.uniform(0.1, 0.25, size=M)
    return t, y, dy, f_true


class TestFP16FastChi2:
    def test_native_fp16_query(self):
        flag = _load_library().tlsh_using_native_fp16()
        assert flag in (0, 1)

    @pytest.mark.parametrize("backend", ["pswf43", "pswf21", "lra"])
    @pytest.mark.parametrize("solver", ["levinson", "zohar", "bareiss", "ldlt", "svd"])
    def test_agreement_with_double_across_backends_and_solvers(self, backend, solver):
        t, y, dy, f_true = _make_signal(M=300)
        Nf, df, f0 = 256, 0.005, 0.01
        p_d = tls.power(Nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=2)
        p_h_f32 = tlsh.power(Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, solver=solver, nterms=2)
        p_h_f64 = tlsh.power(Nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=2)

        assert p_h_f32.dtype == np.float32
        assert p_h_f64.dtype == np.float32
        assert p_h_f32.shape == (Nf,)

        freqs = f0 + df * np.arange(Nf)
        assert abs(freqs[int(np.nanargmax(p_h_f64))] - f_true) <= df
        valid = np.isfinite(p_d) & np.isfinite(p_h_f64)
        assert np.mean(valid) > 0.95
        assert np.max(np.abs(p_h_f64[valid] - p_d[valid])) < 1.5e-2
        assert np.median(np.abs(p_h_f64[valid] - p_d[valid])) < 5e-3
        assert np.max(np.abs(p_h_f32[valid] - p_h_f64[valid])) < 5e-3

    @pytest.mark.parametrize("spreading", ["minmax", "piecewise"])
    @pytest.mark.parametrize("vectorize", ["depth", "breadth"])
    @pytest.mark.parametrize("normalization", ["standard", "anova", "nll"])
    def test_options_and_normalizations(self, spreading, vectorize, normalization):
        t, y, dy, _ = _make_signal(M=250)
        Nf, df, f0 = 256, 0.005, 0.01
        p = tlsh.power(
            Nf,
            df,
            f0,
            t,
            y,
            dy,
            backend="pswf43",
            solver="levinson",
            nterms=3,
            normalization=normalization,
            spreading=spreading,
            vectorize=vectorize,
        )
        p_ref = tls.power(
            Nf,
            df,
            f0,
            t,
            y,
            dy,
            backend="pswf43",
            solver="levinson",
            nterms=3,
            normalization=normalization,
        )
        assert p.dtype == np.float32
        valid = np.isfinite(p) & np.isfinite(p_ref)
        assert np.mean(valid) > 0.95
        assert int(np.nanargmax(p)) == int(np.nanargmax(p_ref))

    def test_autopower_and_get_peaks(self):
        t, y, dy, f_true = _make_signal(M=300)
        freqs, power = tlsh.autopower(t, y.astype(np.float32), dy.astype(np.float32), fmax=2.0, nterms=2)
        assert freqs.dtype == np.float64
        assert power.dtype == np.float32
        peaks = get_peaks(freqs, power, num_peaks=3)
        assert len(peaks) >= 1
        assert abs(peaks[0].freq - f_true) < 0.02

    def test_extreme_scale_invariance_fp64_preprocessing(self):
        t, y, dy, _ = _make_signal(M=250)
        Nf, df, f0 = 256, 0.005, 0.01
        p_base = tlsh.power(Nf, df, f0, t, y, dy, nterms=2)
        for scale in (1e20, 1e-20):
            p_y = tlsh.power(Nf, df, f0, t, y * scale, dy, nterms=2)
            p_dy = tlsh.power(Nf, df, f0, t, y, dy * scale, nterms=2)
            p_both = tlsh.power(Nf, df, f0, t, y * scale + 1e6 * scale, dy * scale, nterms=2)
            np.testing.assert_allclose(p_y, p_base, rtol=1e-3, atol=1e-3, equal_nan=True)
            np.testing.assert_allclose(p_dy, p_base, rtol=1e-3, atol=1e-3, equal_nan=True)
            np.testing.assert_allclose(p_both, p_base, rtol=2e-3, atol=2e-3, equal_nan=True)

    def test_autonan_and_condition_bounds_ill_conditioned(self):
        rng = np.random.default_rng(7)
        M = 120
        t = (np.arange(M) % 20).astype(np.float64) + 0.02 * (np.arange(M) // 20 - 2.5)
        y = np.sin(2.0 * np.pi * 0.23 * t) + 0.1 * rng.normal(size=M)
        dy = np.full(M, 0.1, dtype=np.float64)
        Nf, df, f0 = 256, 0.01, 0.05

        p_nan = tlsh.power(Nf, df, f0, t, y, dy, nterms=3, autonan=True)
        p_cond, cond = tlsh.power(Nf, df, f0, t, y, dy, nterms=3, autonan=False)
        assert cond.dtype == np.float32
        assert p_cond.dtype == np.float32
        high = cond > 500.0
        assert np.any(high)
        assert np.all(np.isnan(p_nan[high]))


class TestFP16IHS:
    @pytest.mark.parametrize("backend", ["pswf43", "pswf21", "lra"])
    @pytest.mark.parametrize("normalization", ["raw", "default", "nll"])
    @pytest.mark.parametrize("vectorize", ["depth", "breadth"])
    def test_ihsh_power_and_autopower(self, backend, normalization, vectorize):
        t, y, dy, f_true = _make_signal(M=300)
        Nf, df, f0 = 256, 0.005, 0.01
        p_h = ihsh.power(Nf, df, f0, t, y, dy, backend=backend, nterms=2, normalization=normalization, vectorize=vectorize)
        p_h_f32 = ihsh.power(
            Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, nterms=2, normalization=normalization, vectorize=vectorize
        )
        p_d = ihs.power(Nf, df, f0, t, y, dy, backend=backend, nterms=2, normalization=normalization)

        assert p_h.dtype == np.float32
        assert p_h_f32.dtype == np.float32
        assert np.all(np.isfinite(p_h))
        assert int(np.argmax(p_h)) == int(np.argmax(p_d))
        np.testing.assert_allclose(p_h, p_h_f32, rtol=1e-2, atol=1e-2)

        freqs, p_auto = ihsh.autopower(t, y, dy, fmax=2.0, nterms=2, backend=backend, normalization=normalization, vectorize=vectorize)
        assert freqs.dtype == np.float64
        assert p_auto.dtype == np.float32
        assert abs(freqs[int(np.argmax(p_auto))] - f_true) < 0.02
