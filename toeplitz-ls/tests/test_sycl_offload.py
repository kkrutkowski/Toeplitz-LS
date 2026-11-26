"""Tests for the SYCL GPU-offloaded periodogram wrappers in toeplitz_ls.offload.sycl."""

from __future__ import annotations

import concurrent.futures
import inspect

import numpy as np
import pytest

import toeplitz_ls as cpu
from toeplitz_ls.offload.sycl import (
    SyclDevice,
    ihs,
    ihsdf,
    ihsf,
    ihsh,
    list_devices,
    tls,
    tlsdf,
    tlsf,
    tlsh,
)


def _make_signal(M: int = 320, seed: int = 42, span: float = 40.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, size=M)).astype(np.float64)
    f_true = 0.37
    y = (
        2.0 * np.sin(2.0 * np.pi * f_true * t)
        + 0.7 * np.cos(2.0 * np.pi * 2.0 * f_true * t + 0.4)
        + 0.3 * np.sin(2.0 * np.pi * 3.0 * f_true * t - 0.2)
        + 0.15 * rng.normal(size=M)
    )
    dy = rng.uniform(0.1, 0.25, size=M)
    return t, y, dy, f_true


def _get_device_id() -> int:
    devices = list_devices(print_table=False)
    assert len(devices) >= 1, "No SYCL devices found"
    return devices[0].id


class TestDeviceDiscoveryAndAPI:
    def test_list_devices_returns_sorted_descriptors(self, capsys):
        devices = list_devices(print_table=True)
        captured = capsys.readouterr().out
        assert len(devices) >= 1
        for idx, dev in enumerate(devices):
            assert isinstance(dev, SyclDevice)
            assert dev.id == idx
            assert isinstance(dev.name, str) and len(dev.name) > 0
            assert f"[{idx}]" in captured
            assert dev.name in captured
        # GPUs must be sorted before non-GPUs
        gpu_flags = [d.is_gpu for d in devices]
        assert gpu_flags == sorted(gpu_flags, reverse=True)

    def test_required_positional_device_signature(self):
        for mod in (tlsh, tlsf, tlsdf, tls, ihsh, ihsf, ihsdf, ihs):
            sig_p = inspect.signature(mod.power)
            sig_a = inspect.signature(mod.autopower)
            p_params = list(sig_p.parameters.values())
            a_params = list(sig_a.parameters.values())
            assert p_params[0].name == "device"
            assert p_params[0].default is inspect.Parameter.empty
            assert a_params[0].name == "device"
            assert a_params[0].default is inspect.Parameter.empty
            # No configurable solver/spreading/vectorize in SYCL offload
            for forbidden in ("solver", "spreading", "vectorize"):
                assert forbidden not in sig_p.parameters
                assert forbidden not in sig_a.parameters

    def test_invalid_device_and_backend_errors(self):
        t, y, dy, _ = _make_signal(M=120)
        dev = _get_device_id()
        with pytest.raises((ValueError, RuntimeError)):
            tlsf.power(9999, 128, 0.01, 0.05, t, y, dy)
        with pytest.raises((ValueError, RuntimeError)):
            tlsf.power(-1, 128, 0.01, 0.05, t, y, dy)
        with pytest.raises(ValueError, match="pswf43.*pswf21|backend"):
            tlsf.power(dev, 128, 0.01, 0.05, t, y, dy, backend="lra")
        with pytest.raises(ValueError, match="pswf43.*pswf21|backend"):
            ihsf.power(dev, 128, 0.01, 0.05, t, y, dy, backend="lra")


class TestFastChi2Parity:
    @pytest.mark.parametrize("nterms", [1, 2, 3, 5, 8, 13])
    @pytest.mark.parametrize("backend", ["pswf43", "pswf21"])
    def test_fp64_fp32_df_fp16_parity_vs_cpu(self, nterms, backend):
        dev = _get_device_id()
        t, y, dy, f_true = _make_signal(M=350, seed=100 + nterms)
        Nf, df, f0 = 600, 0.004, 0.02
        solver_kw = {"solver": "levinson"} if nterms > 1 else {}

        p_cpu_d = cpu.tls.power(Nf, df, f0, t, y, dy, backend=backend, nterms=nterms, **solver_kw)
        p_cpu_f = cpu.tlsf.power(Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, nterms=nterms, **solver_kw)

        p_gpu_d = tls.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms)
        p_gpu_df = tlsdf.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms)
        p_gpu_f = tlsf.power(dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, nterms=nterms)
        p_gpu_h = tlsh.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms)

        assert p_gpu_d.dtype == np.float64
        assert p_gpu_df.dtype == np.float64
        assert p_gpu_f.dtype == np.float32
        assert p_gpu_h.dtype == np.float32

        # FP64 parity (allow cond-amplified rounding at sub-Rayleigh bins for high d)
        tol_d = (1e-11 if nterms <= 5 else 5e-8) if backend == "pswf43" else 1e-6
        np.testing.assert_allclose(p_gpu_d, p_cpu_d, rtol=tol_d, atol=tol_d, equal_nan=True)

        # DF parity vs CPU FP64
        tol_df = 1e-9 if nterms <= 5 else (1e-7 if nterms <= 8 else 5e-6)
        if backend == "pswf21":
            tol_df = max(tol_df, 1e-6)
        np.testing.assert_allclose(p_gpu_df, p_cpu_d, rtol=tol_df, atol=tol_df, equal_nan=True)

        # FP32 parity vs CPU FP32
        tol_f = 5e-5 if nterms <= 8 else 5e-4
        if backend == "pswf21":
            tol_f = max(tol_f, 1e-3)
        np.testing.assert_allclose(p_gpu_f, p_cpu_f, rtol=tol_f, atol=tol_f, equal_nan=True)

        # FP16 parity vs CPU FP64 (skip high d where FP16 Levinson naturally saturates condition)
        if nterms <= 5:
            valid = np.isfinite(p_gpu_h) & np.isfinite(p_cpu_d)
            assert np.mean(valid) > 0.90
            assert np.median(np.abs(p_gpu_h[valid] - p_cpu_d[valid])) < 5e-3
            freqs = f0 + df * np.arange(Nf)
            assert abs(freqs[int(np.nanargmax(p_gpu_h))] - f_true) <= 2 * df

    @pytest.mark.parametrize("normalization", ["standard", "anova", "nll", "regularized"])
    @pytest.mark.parametrize("nterms", [1, 3])
    @pytest.mark.parametrize("use_dy", [False, True])
    def test_all_normalizations_and_dy(self, normalization, nterms, use_dy):
        dev = _get_device_id()
        t, y, dy, _ = _make_signal(M=280, seed=77)
        dy_in = dy if use_dy else None
        Nf, df, f0 = 512, 0.005, 0.03
        solver_kw = {"solver": "levinson"} if nterms > 1 else {}

        p_cpu_d = cpu.tls.power(Nf, df, f0, t, y, dy_in, backend="pswf43", nterms=nterms, normalization=normalization, **solver_kw)
        p_gpu_d = tls.power(dev, Nf, df, f0, t, y, dy_in, backend="pswf43", nterms=nterms, normalization=normalization)
        p_gpu_df = tlsdf.power(dev, Nf, df, f0, t, y, dy_in, backend="pswf43", nterms=nterms, normalization=normalization)
        p_gpu_f = tlsf.power(
            dev,
            Nf,
            df,
            f0,
            t,
            y.astype(np.float32),
            None if dy_in is None else dy_in.astype(np.float32),
            backend="pswf43",
            nterms=nterms,
            normalization=normalization,
        )
        p_gpu_h = tlsh.power(dev, Nf, df, f0, t, y, dy_in, backend="pswf43", nterms=nterms, normalization=normalization)

        np.testing.assert_allclose(p_gpu_d, p_cpu_d, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(p_gpu_df, p_cpu_d, rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(p_gpu_f, p_cpu_d, rtol=1e-4, atol=1e-4, equal_nan=True)
        valid = np.isfinite(p_gpu_h) & np.isfinite(p_cpu_d)
        assert np.mean(valid) > 0.95
        assert int(np.nanargmax(p_gpu_h)) == int(np.nanargmax(p_cpu_d))

    def test_autonan_and_condition_bounds(self):
        dev = _get_device_id()
        rng = np.random.default_rng(7)
        M = 120
        t = (np.arange(M) % 20).astype(np.float64) + 0.02 * (np.arange(M) // 20 - 2.5)
        y = np.sin(2.0 * np.pi * 0.23 * t) + 0.1 * rng.normal(size=M)
        dy = np.full(M, 0.1, dtype=np.float64)
        Nf, df, f0 = 256, 0.01, 0.05

        for mod, thresh, out_dtype in [
            (tlsh, 500.0, np.float32),
            (tlsf, 5e5, np.float32),
            (tlsdf, 1e13, np.float64),
            (tls, 1e13, np.float64),
        ]:
            p_nan = mod.power(dev, Nf, df, f0, t, y, dy, nterms=3, autonan=True)
            p_raw, cond = mod.power(dev, Nf, df, f0, t, y, dy, nterms=3, autonan=False)
            assert p_nan.dtype == out_dtype
            assert p_raw.dtype == out_dtype
            assert cond.dtype == out_dtype
            assert np.all(cond >= 1.0 - 1e-3)
            high = cond > thresh
            if np.any(high):
                assert np.all(np.isnan(p_nan[high]))
            low = cond < thresh * 0.5
            assert np.any(low)
            np.testing.assert_allclose(p_nan[low], p_raw[low], rtol=1e-6, atol=1e-6)

    def test_multi_block_and_tail_grid_lengths(self):
        """Test Nf spanning multiple subgrid blocks and partial tail blocks."""
        dev = _get_device_id()
        t, y, dy, _ = _make_signal(M=400, seed=19, span=120.0)
        for Nf in (17, 511, 513, 2500, 12345):
            df, f0 = 0.001, 0.005
            p_cpu = cpu.tlsf.power(Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), nterms=3, solver="levinson")
            p_gpu = tlsf.power(dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), nterms=3)
            assert p_gpu.shape == (Nf,)
            np.testing.assert_allclose(p_gpu, p_cpu, rtol=5e-5, atol=5e-5, equal_nan=True)

    def test_autopower_matches_cpu_grid_and_values(self):
        dev = _get_device_id()
        t, y, dy, f_true = _make_signal(M=300, seed=55, span=60.0)
        for cmod, gmod, tol in [
            (cpu.tls, tls, 1e-11),
            (cpu.tls, tlsdf, 1e-9),
            (cpu.tlsf, tlsf, 5e-5),
            (cpu.tlsh, tlsh, 1e-2),
        ]:
            fc, pc = cmod.autopower(t, y, dy, fmax=2.0, nterms=3)
            fg, pg = gmod.autopower(dev, t, y, dy, fmax=2.0, nterms=3)
            np.testing.assert_array_equal(fg, fc)
            valid = np.isfinite(pc) & np.isfinite(pg)
            assert np.mean(valid) > 0.95
            assert np.max(np.abs(pg[valid] - pc[valid])) < tol
            assert abs(fg[int(np.nanargmax(pg))] - f_true) < 0.02

    def test_fp16_extreme_scale_invariance(self):
        dev = _get_device_id()
        t, y, dy, _ = _make_signal(M=250)
        Nf, df, f0 = 256, 0.005, 0.01
        p_base = tlsh.power(dev, Nf, df, f0, t, y, dy, nterms=2)
        for scale in (1e20, 1e-20):
            p_both = tlsh.power(dev, Nf, df, f0, t, y * scale + 1e6 * scale, dy * scale, nterms=2)
            np.testing.assert_allclose(p_both, p_base, rtol=2e-3, atol=2e-3, equal_nan=True)


class TestIHSParity:
    @pytest.mark.parametrize("normalization", ["default", "raw", "nll"])
    @pytest.mark.parametrize("nterms", [1, 3, 5])
    @pytest.mark.parametrize("backend", ["pswf43", "pswf21"])
    def test_ihs_all_precisions_vs_cpu(self, normalization, nterms, backend):
        dev = _get_device_id()
        t, y, dy, _ = _make_signal(M=320, seed=200 + nterms)
        Nf, df, f0 = 512, 0.005, 0.01

        p_cpu_d = cpu.ihs.power(Nf, df, f0, t, y, dy, backend=backend, nterms=nterms, normalization=normalization)
        p_cpu_f = cpu.ihsf.power(Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, nterms=nterms, normalization=normalization)

        p_gpu_d = ihs.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms, normalization=normalization)
        p_gpu_df = ihsdf.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms, normalization=normalization)
        p_gpu_f = ihsf.power(dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), backend=backend, nterms=nterms, normalization=normalization)
        p_gpu_h = ihsh.power(dev, Nf, df, f0, t, y, dy, backend=backend, nterms=nterms, normalization=normalization)

        tol_d = 1e-10 if backend == "pswf43" else 1e-5
        np.testing.assert_allclose(p_gpu_d, p_cpu_d, rtol=tol_d, atol=tol_d)
        tol_df = 1e-8 if backend == "pswf43" else 1e-5
        np.testing.assert_allclose(p_gpu_df, p_cpu_d, rtol=tol_df, atol=tol_df)
        tol_f = 1e-4 if backend == "pswf43" else 1e-3
        np.testing.assert_allclose(p_gpu_f, p_cpu_f, rtol=tol_f, atol=tol_f)

        assert np.all(np.isfinite(p_gpu_h))
        assert int(np.argmax(p_gpu_h)) == int(np.argmax(p_cpu_d))
        np.testing.assert_allclose(p_gpu_h, p_cpu_d, rtol=2e-2, atol=2e-2)

    def test_ihs_undersampling_warning(self):
        dev = _get_device_id()
        t, y, dy, _ = _make_signal(M=30, seed=11)
        with pytest.warns(RuntimeWarning):
            ihsf.power(dev, 128, 0.01, 0.05, t, y, dy, nterms=4, normalization="default")


class TestThreadPoolConcurrency:
    def test_threadpool_heterogeneous_stress(self):
        dev = _get_device_id()
        curves = []
        for idx in range(48):
            M = 80 + (idx % 9) * 110
            span = 30.0 + (idx % 5) * 40.0
            t, y, dy, _ = _make_signal(M=M, seed=500 + idx, span=span)
            d = (1, 2, 3, 5)[idx % 4]
            Nf = 200 + (idx % 7) * 700
            curves.append((idx, t, y, dy, d, Nf))

        # Reference sequential results
        ref = {}
        for idx, t, y, dy, d, Nf in curves:
            ref[idx] = tlsf.power(dev, Nf, 0.003, 0.01, t, y.astype(np.float32), dy.astype(np.float32), nterms=d)

        # Concurrent execution across 15 worker threads (and mixed precisions/families)
        def worker(item):
            idx, t, y, dy, d, Nf = item
            p = tlsf.power(dev, Nf, 0.003, 0.01, t, y.astype(np.float32), dy.astype(np.float32), nterms=d)
            if idx % 6 == 0:
                _ = tlsh.power(dev, Nf, 0.003, 0.01, t, y, dy, nterms=min(d, 3))
            if idx % 8 == 0:
                _ = ihsf.power(dev, Nf, 0.003, 0.01, t, y.astype(np.float32), dy.astype(np.float32), nterms=d)
            return idx, p

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
            results = dict(ex.map(worker, curves))

        for idx in range(len(curves)):
            # GPU FP32 atomic spreading summation order can vary by ~1 ULP across runs
            np.testing.assert_allclose(results[idx], ref[idx], rtol=1e-5, atol=1e-6, equal_nan=True)


class TestHardwareBudgetAndOddM:
    def test_memory_budget_rejects_oversized_grid(self):
        devices = list_devices(print_table=False)
        dev = devices[0]
        t, y, dy, _ = _make_signal(M=64, seed=77)
        # Choose Nf so double-precision output alone exceeds 75% of total device memory
        oversized_nf = int((dev.global_mem_size // 8) + 1024)
        oversized_nf = min(oversized_nf, 2_000_000_000)
        with pytest.raises(RuntimeError, match="memory budget|allocation"):
            tls.power(dev.id, oversized_nf, 1e-7, 1e-7, t, y, dy, nterms=1)

    def test_odd_m_multiblock_strip_recurrence_parity(self):
        dev = _get_device_id()
        # Odd M (351) + large Nf (25000 -> B_total >= 16) exercises padded WG sizing,
        # K_b=8 subgrid strip-recurrence, and fused single-harmonic IHS.
        t, y, dy, _ = _make_signal(M=351, seed=313, span=120.0)
        Nf, df, f0 = 25000, 0.0015, 0.005

        for d in (1, 3):
            solver_kw = {"solver": "levinson"} if d > 1 else {}
            p_cpu = cpu.tls.power(Nf, df, f0, t, y, dy, nterms=d, **solver_kw)
            p_d = tls.power(dev, Nf, df, f0, t, y, dy, nterms=d)
            p_df = tlsdf.power(dev, Nf, df, f0, t, y, dy, nterms=d)
            p_f = tlsf.power(dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), nterms=d)
            p_h = tlsh.power(dev, Nf, df, f0, t, y, dy, nterms=d)

            np.testing.assert_allclose(p_d, p_cpu, rtol=1e-10, atol=1e-10, equal_nan=True)
            np.testing.assert_allclose(p_df, p_cpu, rtol=1e-8, atol=1e-8, equal_nan=True)
            np.testing.assert_allclose(p_f, p_cpu, rtol=1e-3, atol=1e-3, equal_nan=True)
            assert int(np.nanargmax(p_h)) == int(np.nanargmax(p_cpu))

        # Fused d=1 IHS across all 4 precisions & normalizations
        for norm in ("default", "raw", "nll"):
            p_cpu = cpu.ihs.power(Nf, df, f0, t, y, dy, nterms=1, normalization=norm)
            p_d = ihs.power(dev, Nf, df, f0, t, y, dy, nterms=1, normalization=norm)
            p_df = ihsdf.power(dev, Nf, df, f0, t, y, dy, nterms=1, normalization=norm)
            p_f = ihsf.power(dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), nterms=1, normalization=norm)
            p_h = ihsh.power(dev, Nf, df, f0, t, y, dy, nterms=1, normalization=norm)
            np.testing.assert_allclose(p_d, p_cpu, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(p_df, p_cpu, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(p_f, p_cpu, rtol=1e-3, atol=1e-3)
            assert int(np.argmax(p_h)) == int(np.argmax(p_cpu))

    def test_omp_cpu_device_parity(self):
        devices = list_devices(print_table=False)
        omp_devs = [d for d in devices if not d.is_gpu and d.backend == "omp"]
        if not omp_devs:
            pytest.skip("No SYCL OpenMP CPU device available")
        cpu_dev = omp_devs[0].id
        t, y, dy, _ = _make_signal(M=257, seed=99)
        Nf, df, f0 = 1024, 0.005, 0.025
        p_ref = cpu.tls.power(Nf, df, f0, t, y, dy, nterms=3, solver="levinson")
        p_d = tls.power(cpu_dev, Nf, df, f0, t, y, dy, nterms=3)
        p_df = tlsdf.power(cpu_dev, Nf, df, f0, t, y, dy, nterms=3)
        p_f = tlsf.power(cpu_dev, Nf, df, f0, t, y.astype(np.float32), dy.astype(np.float32), nterms=3)
        np.testing.assert_allclose(p_d, p_ref, rtol=1e-10, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(p_df, p_ref, rtol=1e-8, atol=1e-8, equal_nan=True)
        np.testing.assert_allclose(p_f, p_ref, rtol=1e-3, atol=1e-3, equal_nan=True)
