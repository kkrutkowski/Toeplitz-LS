"""SYCL GPU-offloaded incoherent harmonic sum (IHS) periodogram wrappers.

Mirrors :mod:`toeplitz_ls._ihs` with the required SYCL target ``device`` ID
as the first positional argument.
"""

from __future__ import annotations

import ctypes

import numpy as np

from ..._fastchi2 import (
    _auto_grid,
    _check_grid,
    _check_int,
    _check_observation_count,
    _prepare_half_inputs,
    _prepare_numpy_inputs,
)
from ..._ihs import _check_ihs_normalization, _raw_neff, _warn_if_ihs_undersampled
from ._fastchi2 import _check_device_id, _sycl_backend_id
from ._lib import load_library, raise_status


def _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization):
    _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
    out = np.empty(Nf, dtype=np.float32)
    lib = load_library()
    if use_double:
        status = lib.tls_sycl_h_ihs_d(
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
        )
        raise_status("tls_sycl_h_ihs_d", status)
    else:
        status = lib.tls_sycl_h_ihs(
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
        )
        raise_status("tls_sycl_h_ihs", status)
    return out


def _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization):
    _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
    out = np.empty(Nf, dtype=np.float32)
    status = load_library().tls_sycl_f_ihs(
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
    )
    raise_status("tls_sycl_f_ihs", status)
    return out


def _run_ihs_double_prepared(fn_name, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization):
    _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
    out = np.empty(Nf, dtype=np.float64)
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
    )
    raise_status(fn_name, status)
    return out


class ihsh:
    """Half-precision (IEEE binary16) SYCL GPU IHS periodogram wrapper."""

    @staticmethod
    def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(t, y, dy)
        _check_observation_count(t_arr.size, nterms)
        return _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization)

    @staticmethod
    def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic half-precision IHS frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(x, y, dy)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization)
        return frequency, result


class ihsf:
    """Single-precision (FP32) SYCL GPU IHS periodogram wrapper."""

    @staticmethod
    def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        return _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

    @staticmethod
    def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic single-precision IHS frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
        return frequency, result


class ihsdf:
    """Dekker double-float (2xFP32) SYCL GPU IHS periodogram wrapper (float64 I/O)."""

    @staticmethod
    def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        return _run_ihs_double_prepared("tls_sycl_df_ihs", device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

    @staticmethod
    def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic double-float IHS frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_ihs_double_prepared("tls_sycl_df_ihs", device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
        return frequency, result


class ihs:
    """Native double-precision (FP64) SYCL GPU IHS periodogram wrapper."""

    @staticmethod
    def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        return _run_ihs_double_prepared("tls_sycl_d_ihs", device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

    @staticmethod
    def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic double-precision IHS frequency and power arrays offloaded to ``device``."""
        device = _check_device_id(device)
        normalization = _check_ihs_normalization(normalization)
        backend_id = _sycl_backend_id(backend)
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = float(np.max(t_arr) - np.min(t_arr))
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = _run_ihs_double_prepared("tls_sycl_d_ihs", device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
        return frequency, result


__all__ = ["ihsh", "ihsf", "ihsdf", "ihs"]
