"""ctypes wrappers for the incoherent harmonic sum (IHS) periodogram.

The IHS path is a deliberately simpler alternative to the FastChi2/AoVMH
periodogram in :mod:`toeplitz_ls._fastchi2`.  It evaluates one NuFFT per
harmonic and accumulates the squared magnitudes directly, so it never builds
or solves the Toeplitz normal equations.

Two statistics are available through ``normalization=``:

``standard`` (default)
    Sum of the uncorrelated per-harmonic coefficient-of-determination
    estimates, ``2 * sum_k |Z_k(f)|^2 / (sum_m w_m * chi2_ref)``.
``nll``
    The IHS negative log false-alarm probability.  Each harmonic power is
    normalized to the Rayleigh scale and passed through Wilkie's finite-sample
    correction before the harmonics are summed.  The sum is then passed
    through the Erlang correction ``sum - ln(sum_{k=0}^{d-1} sum^k / k!)``.

Unlike the FastChi2 wrappers these modules do not accept ``solver`` or
``autonan`` and do not return a condition array: IHS has no linear solve and
no Toeplitz condition bound.
"""

from __future__ import annotations

import ctypes

import numpy as np

from ._fastchi2 import (
    _BACKENDS,
    DD,
    _as_dd_array,
    _as_mpf_list,
    _auto_grid,
    _backend_id,
    _check_grid,
    _check_int,
    _check_observation_count,
    _dd_to_mpf,
    _load_library,
    _prepare_numpy_inputs,
    _raise_status,
    _require_mpmath,
)

_IHS_NORMALIZATIONS = {"standard": 0, "nll": 2}


def _check_ihs_normalization(normalization):
    if isinstance(normalization, str):
        key = normalization.strip().lower()
        if key in _IHS_NORMALIZATIONS:
            return _IHS_NORMALIZATIONS[key]
    if isinstance(normalization, int) and not isinstance(normalization, bool):
        if normalization in _IHS_NORMALIZATIONS.values():
            return normalization
    raise ValueError("normalization must be 'standard' or 'nll'")


class ihsf:
    """Single-precision IHS periodogram wrapper."""

    @staticmethod
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="standard"):
        """Return the IHS power on the grid ``f0 + df * arange(Nf)``.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        normalization = _check_ihs_normalization(normalization)
        backend = _backend_id(backend)
        min_Nf = 32 if backend == _BACKENDS["lra"] else 16
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=min_Nf)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        out = np.empty(Nf, dtype=np.float32)

        status = _load_library().tlsf_ihs(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend,
            normalization,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        _raise_status("tlsf_ihs", status)
        return out

    @staticmethod
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="standard", nterms=3, backend="pswf43"):
        """Return automatic IHS frequency and power arrays.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        delta_t = np.max(t_arr) - np.min(t_arr)
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = ihsf.power(
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend=backend,
            normalization=normalization,
            nterms=nterms,
        )
        return frequency, result


class ihs:
    """Double-precision IHS periodogram wrapper."""

    @staticmethod
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="standard"):
        """Return the IHS power on the grid ``f0 + df * arange(Nf)``.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        normalization = _check_ihs_normalization(normalization)
        backend = _backend_id(backend)
        min_Nf = 32 if backend == _BACKENDS["lra"] else 8
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=min_Nf)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        out = np.empty(Nf, dtype=np.float64)

        status = _load_library().tls_ihs(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend,
            normalization,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        _raise_status("tls_ihs", status)
        return out

    @staticmethod
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="standard", nterms=3, backend="pswf43"):
        """Return automatic double-precision IHS frequency and power arrays.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        nterms = _check_int("nterms", nterms, 1)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = np.max(t_arr) - np.min(t_arr)
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = ihs.power(
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend=backend,
            normalization=normalization,
            nterms=nterms,
        )
        return frequency, result


class ihsdd:
    """Double-double IHS periodogram wrapper returning ``mpmath`` values."""

    @staticmethod
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="standard"):
        """Return high-precision IHS power.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        mp = _require_mpmath()
        normalization = _check_ihs_normalization(normalization)
        Nf = _check_int("Nf", Nf, 1)
        nterms = _check_int("nterms", nterms, 1)
        df_mpf = mp.mpf(df)
        f0_mpf = mp.mpf(f0)
        if not mp.isfinite(df_mpf):
            raise ValueError("df must be finite")
        if not mp.isfinite(f0_mpf):
            raise ValueError("f0 must be finite")
        if df_mpf <= 0:
            raise ValueError("df must be positive")
        if f0_mpf < 0:
            raise ValueError("f0 must be non-negative")

        backend = _backend_id(backend)
        t_mpf = _as_mpf_list(t, "t")
        y_mpf = _as_mpf_list(y, "y")
        if dy is None:
            dy_mpf = [mp.mpf(1) for _ in y_mpf]
        else:
            dy_mpf = _as_mpf_list(dy, "dy")

        if len(t_mpf) != len(y_mpf) or len(t_mpf) != len(dy_mpf):
            raise ValueError("t, y, and dy must have the same length")
        if any(value <= 0 for value in dy_mpf):
            raise ValueError("dy entries must be positive")
        _check_observation_count(len(t_mpf), nterms)

        t_dd = _as_dd_array(t_mpf)
        y_dd = _as_dd_array(y_mpf)
        dy_dd = _as_dd_array(dy_mpf)
        out = (DD * Nf)()

        status = _load_library().tlsdd_ihs(
            t_dd,
            y_dd,
            dy_dd,
            len(t_mpf),
            float(f0_mpf),
            float(df_mpf),
            Nf,
            nterms,
            backend,
            normalization,
            out,
        )
        _raise_status("tlsdd_ihs", status)
        return [_dd_to_mpf(out[i]) for i in range(Nf)]

    @staticmethod
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="standard", nterms=3, backend="pswf43"):
        """Return automatic high-precision IHS frequency and power lists.

        ``normalization`` accepts ``'standard'`` (default) or ``'nll'``.
        """
        mp = _require_mpmath()
        nterms = _check_int("nterms", nterms, 1)
        t_mpf = _as_mpf_list(x, "x")
        y_mpf = _as_mpf_list(y, "y")
        if dy is None:
            dy_mpf = [mp.mpf(1) for _ in y_mpf]
        else:
            dy_mpf = _as_mpf_list(dy, "dy")

        if len(t_mpf) != len(y_mpf) or len(t_mpf) != len(dy_mpf):
            raise ValueError("x, y, and dy must have the same length")
        if any(value <= 0 for value in dy_mpf):
            raise ValueError("dy entries must be positive")
        _check_observation_count(len(t_mpf), nterms)

        delta_t = max(t_mpf) - min(t_mpf)
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        df_mpf = mp.mpf(df)
        f0_mpf = mp.mpf(f0)
        frequency = [f0_mpf + i * df_mpf for i in range(Nf)]
        result = ihsdd.power(
            Nf,
            df_mpf,
            f0_mpf,
            t_mpf,
            y_mpf,
            dy_mpf,
            backend=backend,
            normalization=normalization,
            nterms=nterms,
        )
        return frequency, result


__all__ = ["ihsf", "ihs", "ihsdd"]
