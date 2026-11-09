"""ctypes wrappers for the incoherent harmonic sum (IHS) periodogram.

The IHS path is a simplified formulation of a similar periodogram.  It assumes
the harmonic frequencies are independent (a regression from the standard
least-squares periodogram) but not orthogonal.  It evaluates one NuFFT per
harmonic and accumulates the squared magnitudes directly, so it never builds
or solves the Toeplitz normal equations.  In practice it is usually slightly
less accurate than FastChi2, but at least 3x less computationally expensive.

Three statistics are available through ``normalization=``:

``raw``
    Direct display statistic from the per-harmonic coefficient-of-determination
    estimates.  It is not a calibrated false-alarm probability; for
    ``nterms > 1`` this is a sum, not a bounded R^2, and may exceed 1.
``nll``
    Negative log false-alarm probability with Wilkie's (1983) finite-sample
    correction for the Rayleigh Z test applied to each harmonic, followed by
    the Erlang correction of the harmonic sum.  Use this when a reliable
    significance estimate is needed.
``default`` (the default)
    The same initial formula and Erlang correction as ``nll``, but without the
    finite-sample Wilkie correction.  This still produces NLL estimates, just
    less accurately.

Unlike the FastChi2 wrappers these modules do not accept ``solver`` or
``autonan`` and do not return a condition array: IHS has no linear solve and
no Toeplitz condition bound.

The approximation is not suitable when ``nterms**4 > Neff``; the Python
wrappers emit a :class:`RuntimeWarning` in that case.  The warning uses the
raw-weight ``Neff`` from the FastChi2/TLS path, not the weighted-residual
``Neff`` used inside the IHS NLL.
"""

from __future__ import annotations

import ctypes
import warnings

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

_IHS_NORMALIZATIONS = {
    "raw": 0,
    "nll": 2,
    "default": 4,
    # Legacy alias, intentionally undocumented:
    "standard": 0,
}


def _check_ihs_normalization(normalization):
    if isinstance(normalization, str):
        key = normalization.strip().lower()
        if key in _IHS_NORMALIZATIONS:
            return _IHS_NORMALIZATIONS[key]
    if isinstance(normalization, int) and not isinstance(normalization, bool):
        if normalization in _IHS_NORMALIZATIONS.values():
            return normalization
    raise ValueError("normalization must be 'raw', 'nll', or 'default'")


def _raw_neff(dy):
    """Effective number of measurements from the raw weights, 1 / dy^2.

    The IHS warning uses the same raw-weight Neff as the FastChi2/TLS path;
    the NLL itself uses a different Neff built from the weighted y residuals.
    """
    w = 1.0 / np.asarray(dy, dtype=np.float64) ** 2
    return float(w.sum() ** 2 / np.sum(w * w))


def _warn_if_ihs_undersampled(n_eff, nterms):
    """Warn when the IHS approximation is outside its intended regime."""
    if nterms**4 > n_eff:
        warnings.warn(
            "IHS is not suitable when nterms**4 > Neff "
            f"(nterms={nterms}, Neff={n_eff:.6g}); use the "
            "Toeplitz FastChi2 modules (tls/tlsf/tlsdd) or reduce nterms.",
            RuntimeWarning,
            stacklevel=3,
        )


class ihsf:
    """Single-precision IHS periodogram wrapper."""

    @staticmethod
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return the IHS power on the grid ``f0 + df * arange(Nf)``.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
        """
        normalization = _check_ihs_normalization(normalization)
        backend = _backend_id(backend)
        min_Nf = 32 if backend == _BACKENDS["lra"] else 16
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=min_Nf)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
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
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic IHS frequency and power arrays.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
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
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return the IHS power on the grid ``f0 + df * arange(Nf)``.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
        """
        normalization = _check_ihs_normalization(normalization)
        backend = _backend_id(backend)
        min_Nf = 32 if backend == _BACKENDS["lra"] else 8
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=min_Nf)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
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
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic double-precision IHS frequency and power arrays.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
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
    def power(Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
        """Return high-precision IHS power.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
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
        _warn_if_ihs_undersampled(_raw_neff(dy_mpf), nterms)

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
    def autopower(x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
        """Return automatic high-precision IHS frequency and power lists.

        ``normalization`` accepts ``'raw'``, ``'nll'``, or ``'default'``.
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
