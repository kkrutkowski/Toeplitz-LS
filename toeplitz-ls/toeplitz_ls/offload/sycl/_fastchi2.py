"""SYCL GPU-offloaded FastChi2/AoVMH(W) least-squares periodogram wrappers.

Mirrors :mod:`toeplitz_ls._fastchi2` with the required SYCL target ``device`` ID
as the first positional argument and no configurable solver/spreading/vectorize
options (degree == 1 uses GLS; degree > 1 uses Hermitian Toeplitz Levinson
recursion on the GPU).

Available precision classes:
- ``tlsh``: IEEE binary16 (FP16) compute with ``float32`` outputs
- ``tlsf``: single-precision (FP32) compute and outputs
- ``tlsdf``: Dekker double-float (2xFP32, ~47-48 bit mantissa) with ``float64`` I/O
- ``tls``: native double-precision (FP64) compute and outputs
"""

from __future__ import annotations

import ctypes

import numpy as np

from ..._fastchi2 import (
    _auto_grid,
    _check_grid,
    _check_int,
    _check_normalization,
    _check_observation_count,
    _prepare_half_inputs,
    _prepare_numpy_inputs,
)
from ._lib import load_library, raise_status

_SYCL_BACKENDS = {"pswf43": 1, "pswf21": 2}


def _sycl_backend_id(backend) -> int:
    if isinstance(backend, str):
        key = backend.strip().lower()
        if key in _SYCL_BACKENDS:
            return _SYCL_BACKENDS[key]
    raise ValueError("SYCL offload backend must be 'pswf43' or 'pswf21'")


def _check_device_id(device) -> int:
    return _check_int("device", device, 0)


def _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan):
    out = np.empty(Nf, dtype=np.float32)
    cond = None if autonan else np.empty(Nf, dtype=np.float32)
    cond_ptr = ctypes.POINTER(ctypes.c_float)() if cond is None else cond.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib = load_library()
    if use_double:
        status = lib.tls_sycl_h_fastchi2_d(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cond_ptr,
        )
        raise_status("tls_sycl_h_fastchi2_d", status)
    else:
        status = lib.tls_sycl_h_fastchi2(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cond_ptr,
        )
        raise_status("tls_sycl_h_fastchi2", status)
    if autonan:
        return out
    return out, cond


def _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan):
    out = np.empty(Nf, dtype=np.float32)
    cond = None if autonan else np.empty(Nf, dtype=np.float32)
    cond_ptr = ctypes.POINTER(ctypes.c_float)() if cond is None else cond.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    status = load_library().tls_sycl_f_fastchi2(
        t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        t_arr.size,
        f0,
        df,
        Nf,
        nterms,
        backend_id,
        normalization,
        device,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cond_ptr,
    )
    raise_status("tls_sycl_f_fastchi2", status)
    if autonan:
        return out
    return out, cond


def _run_double_prepared(fn_name, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan):
    out = np.empty(Nf, dtype=np.float64)
    cond = None if autonan else np.empty(Nf, dtype=np.float64)
    cond_ptr = ctypes.POINTER(ctypes.c_double)() if cond is None else cond.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fn = getattr(load_library(), fn_name)
    status = fn(
        t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t_arr.size,
        f0,
        df,
        Nf,
        nterms,
        backend_id,
        normalization,
        device,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cond_ptr,
    )
    raise_status(fn_name, status)
    if autonan:
        return out
    return out, cond


class tlsh:
    """Half-precision (IEEE binary16) SYCL GPU FastChi2 periodogram wrapper."""

    @staticmethod
    def power(
        device,
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(t, y, dy)
        _check_observation_count(t_arr.size, nterms)
        return _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan)

    @staticmethod
    def autopower(
        device,
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        *,
        autonan=True,
    ):
        """Return automatic frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(x, y, dy)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan)
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


class tlsf:
    """Single-precision (FP32) SYCL GPU FastChi2 periodogram wrapper."""

    @staticmethod
    def power(
        device,
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        return _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)

    @staticmethod
    def autopower(
        device,
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        *,
        autonan=True,
    ):
        """Return automatic frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


class tlsdf:
    """Dekker double-float (2xFP32) SYCL GPU FastChi2 periodogram wrapper (float64 I/O)."""

    @staticmethod
    def power(
        device,
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        return _run_double_prepared(
            "tls_sycl_df_fastchi2",
            device,
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend_id,
            nterms,
            normalization,
            autonan,
        )

    @staticmethod
    def autopower(
        device,
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        *,
        autonan=True,
    ):
        """Return automatic frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_double_prepared(
            "tls_sycl_df_fastchi2",
            device,
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend_id,
            nterms,
            normalization,
            autonan,
        )
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


class tls:
    """Native double-precision (FP64) SYCL GPU FastChi2 periodogram wrapper."""

    @staticmethod
    def power(
        device,
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        return _run_double_prepared(
            "tls_sycl_d_fastchi2",
            device,
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend_id,
            nterms,
            normalization,
            autonan,
        )

    @staticmethod
    def autopower(
        device,
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        *,
        autonan=True,
    ):
        """Return automatic frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_double_prepared(
            "tls_sycl_d_fastchi2",
            device,
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend_id,
            nterms,
            normalization,
            autonan,
        )
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


__all__ = ["tlsh", "tlsf", "tlsdf", "tls"]
