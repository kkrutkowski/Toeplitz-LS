"""ctypes wrappers for the FastChi2/AoVMH(W) least-squares periodogram.

The bundled shared library returns the standard normalized power, i.e. the
coefficient of determination R^2. This module can also transform that result
to the asymptotic AOV-style normalization used by ``lcperiod.c``.

The library is loaded with ``ctypes.CDLL``. On CPython, ctypes releases the GIL
while calling functions from CDLL instances, so independent native
``power()``/``autopower()`` computations can run in parallel from a
``ThreadPoolExecutor`` while the C routines are executing.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

try:
    from mpmath import mp as _mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    _mp = None


HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "tls.so"

_BACKENDS = {"pswf43": 1, "pswf21": 2, "lra": 3, "pswf": 1}
_SOLVERS = {"levinson": 1, "zohar": 2, "bareiss": 3, "ldlt": 4}
_NORMALIZATIONS = {"standard", "asymptotic"}
_STATUS_MESSAGES = {
    -1: "invalid argument",
    -2: "invalid backend",
    -3: "allocation failure",
    -4: "degenerate input",
    -5: "invalid solver",
}
_LIB = None


def trim_mantissa(x, bits):
    """Return ``x`` with its binary mantissa truncated to ``bits`` bits."""
    mant, exp = math.frexp(x)
    mant = math.floor(mant * (1 << bits)) / (1 << bits)
    return math.ldexp(mant, exp)


class DD(ctypes.Structure):
    """Double-double value passed to and from the ``tlsdd`` C entry point."""

    _fields_ = [("hi", ctypes.c_double), ("lo", ctypes.c_double)]


def _load_library():
    global _LIB
    if _LIB is not None:
        return _LIB

    if not LIB_PATH.exists():
        raise FileNotFoundError(
            f"{LIB_PATH} does not exist. Build from the repository root with "
            "'make generic' or 'make native' before installing."
        )

    lib = ctypes.CDLL(str(LIB_PATH))

    lib.tlsf_fastchi2.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tlsf_fastchi2.restype = ctypes.c_int

    lib.tls_fastchi2.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.tls_fastchi2.restype = ctypes.c_int

    lib.tlsdd_fastchi2.argtypes = [
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
    ]
    lib.tlsdd_fastchi2.restype = ctypes.c_int

    _LIB = lib
    return lib


def _backend_id(backend):
    if isinstance(backend, str):
        key = backend.lower()
        if key in _BACKENDS:
            return _BACKENDS[key]
    if isinstance(backend, int) and backend in _BACKENDS.values():
        return backend
    raise ValueError("backend must be 'pswf43', 'pswf21', or 'lra'")


def _solver_id(solver):
    if isinstance(solver, str):
        key = solver.lower()
        if key in _SOLVERS:
            return _SOLVERS[key]
    raise ValueError("solver must be 'levinson', 'zohar', 'bareiss', or 'ldlt'")


def _check_normalization(normalization):
    if isinstance(normalization, str):
        key = normalization.lower()
        if key in _NORMALIZATIONS:
            return key
    raise ValueError("normalization must be 'standard' or 'asymptotic'")


def _check_grid(Nf, df, f0, nterms, min_Nf=1):
    Nf = int(Nf)
    nterms = int(nterms)
    df = float(df)
    f0 = float(f0)
    if Nf < min_Nf:
        raise ValueError(f"Nf must be at least {min_Nf}")
    if df <= 0.0:
        raise ValueError("df must be positive")
    if f0 < 0.0:
        raise ValueError("f0 must be non-negative")
    if nterms <= 0:
        raise ValueError("nterms must be positive")
    return Nf, df, f0, nterms


def _auto_grid(delta_t, fmin, fmax, oversampling, nterms):
    delta_t = float(delta_t)
    oversampling = float(oversampling)
    fmax = float(fmax)

    if delta_t <= 0.0:
        raise ValueError("x must span a positive interval")
    if oversampling <= 0.0:
        raise ValueError("oversampling must be positive")
    if fmax <= 0.0:
        raise ValueError("fmax must be positive")

    df = trim_mantissa(1.0 / (oversampling * delta_t * nterms), 5)
    max_frequency = fmax

    if fmin is None:
        first_multiple = math.floor((1.0 / delta_t) / df) + 1
    else:
        fmin = float(fmin)
        if fmin < 0.0:
            raise ValueError("fmin must be non-negative")
        first_multiple = math.ceil((fmin / delta_t) / df)

    f0 = first_multiple * df
    Nf = math.floor((max_frequency - f0) / df) + 1
    if Nf <= 0:
        raise ValueError("automatic frequency grid is empty")

    return Nf, df, f0


def _prepare_numpy_inputs(t, y, dy, y_dtype):
    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=y_dtype)
    if dy is None:
        dy_arr = np.ones_like(y_arr, dtype=y_dtype)
    else:
        dy_arr = np.ascontiguousarray(dy, dtype=y_dtype)

    if t_arr.ndim != 1 or y_arr.ndim != 1 or dy_arr.ndim != 1:
        raise ValueError("t, y, and dy must be one-dimensional")
    if t_arr.size != y_arr.size or t_arr.size != dy_arr.size:
        raise ValueError("t, y, and dy must have the same length")
    if t_arr.size <= 0:
        raise ValueError("t, y, and dy must not be empty")
    if np.any(dy_arr <= 0):
        raise ValueError("dy entries must be positive")

    return t_arr, y_arr, dy_arr


def _check_observation_count(num_observations, nterms):
    min_observations = 2 * nterms + 2
    if num_observations < min_observations:
        raise ValueError(
            f"at least {min_observations} observations are required for nterms={nterms}"
        )


def _asymptotic_dofs(num_observations, nterms):
    d1 = 2 * nterms + 1
    d2 = num_observations - 1 - d1
    if d2 <= 0:
        raise ValueError(
            "asymptotic normalization requires more than 2 * nterms + 2 observations"
        )
    return d1, d2


def _numpy_chi2_ref(y_arr, dy_arr):
    w_arr = np.reciprocal(dy_arr * dy_arr)
    ymean = np.sum(w_arr * y_arr) / np.sum(w_arr)
    yc_arr = y_arr - ymean
    return np.sum(yc_arr * yc_arr * w_arr)


def _normalize_numpy_power(power, normalization, num_observations, nterms, chi2_ref):
    if normalization == "standard":
        return power

    d1, d2 = _asymptotic_dofs(num_observations, nterms)
    dot = power * chi2_ref
    residual = np.maximum(chi2_ref - dot, 1e-32)
    scaled = (d2 * dot) / (d1 * residual)
    return scaled.astype(power.dtype, copy=False)


def _raise_status(name, status):
    if status < 0:
        message = _STATUS_MESSAGES.get(status, f"status {status}")
        raise RuntimeError(f"{name} failed: {message}")


def _require_mpmath():
    if _mp is None:
        raise ImportError("tlsdd requires mpmath to be installed")
    return _mp


def _mpf_to_dd(value):
    mp = _require_mpmath()
    value = mp.mpf(value)
    hi = float(value)
    lo = float(value - mp.mpf(hi))
    return DD(hi, lo)


def _as_mpf_list(values, name):
    mp = _require_mpmath()
    try:
        result = [mp.mpf(value) for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _as_dd_array(values):
    return (DD * len(values))(*(_mpf_to_dd(value) for value in values))


def _dd_to_mpf(value):
    mp = _require_mpmath()
    return mp.mpf(value.hi) + mp.mpf(value.lo)


def _mp_chi2_ref(y_mpf, dy_mpf):
    mp = _require_mpmath()
    weights = [1 / (value * value) for value in dy_mpf]
    weight_sum = mp.fsum(weights)
    ymean = mp.fsum(w * y for w, y in zip(weights, y_mpf)) / weight_sum
    return mp.fsum(w * (y - ymean) ** 2 for w, y in zip(weights, y_mpf))


def _normalize_mpf_power(power, normalization, num_observations, nterms, chi2_ref):
    if normalization == "standard":
        return power

    mp = _require_mpmath()
    d1, d2 = _asymptotic_dofs(num_observations, nterms)
    floor = mp.mpf("1e-32")
    result = []
    for value in power:
        dot = value * chi2_ref
        residual = max(chi2_ref - dot, floor)
        result.append(d2 * dot / (d1 * residual))
    return result


class tlsf:
    """Single-precision FastChi2 periodogram wrapper."""

    @staticmethod
    def power(
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        solver="levinson",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on the grid ``f0 + df * arange(Nf)``."""
        normalization = _check_normalization(normalization)
        backend = _backend_id(backend)
        solver = _solver_id(solver)
        min_Nf = 32 if backend == _BACKENDS["lra"] else 16
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=min_Nf)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        out = np.empty(Nf, dtype=np.float32)
        cond = None if autonan else np.empty(Nf, dtype=np.float32)
        cond_ptr = (
            ctypes.POINTER(ctypes.c_float)()
            if cond is None
            else cond.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        status = _load_library().tlsf_fastchi2(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend,
            solver,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cond_ptr,
        )
        _raise_status("tlsf_fastchi2", status)
        if normalization == "standard":
            power = out
        else:
            power = _normalize_numpy_power(
                out, normalization, t_arr.size, nterms, _numpy_chi2_ref(y_arr, dy_arr)
            )
        if autonan:
            return power
        return power, cond

    @staticmethod
    def autopower(
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        solver="levinson",
        *,
        autonan=True,
    ):
        """Return automatic frequency and power arrays."""
        nterms = int(nterms)
        if nterms <= 0:
            raise ValueError("nterms must be positive")
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
        _check_observation_count(t_arr.size, nterms)
        delta_t = np.max(t_arr) - np.min(t_arr)
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = tlsf.power(
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend=backend,
            solver=solver,
            normalization=normalization,
            nterms=nterms,
            autonan=autonan,
        )
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


class tls:
    """Double-precision FastChi2 periodogram wrapper."""

    @staticmethod
    def power(
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        solver="levinson",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return periodogram power on the grid ``f0 + df * arange(Nf)``."""
        normalization = _check_normalization(normalization)
        backend = _backend_id(backend)
        solver = _solver_id(solver)
        Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        out = np.empty(Nf, dtype=np.float64)
        cond = None if autonan else np.empty(Nf, dtype=np.float64)
        cond_ptr = (
            ctypes.POINTER(ctypes.c_double)()
            if cond is None
            else cond.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        status = _load_library().tls_fastchi2(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dy_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend,
            solver,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cond_ptr,
        )
        _raise_status("tls_fastchi2", status)
        if normalization == "standard":
            power = out
        else:
            power = _normalize_numpy_power(
                out, normalization, t_arr.size, nterms, _numpy_chi2_ref(y_arr, dy_arr)
            )
        if autonan:
            return power
        return power, cond

    @staticmethod
    def autopower(
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        solver="levinson",
        *,
        autonan=True,
    ):
        """Return automatic double-precision frequency and power arrays."""
        nterms = int(nterms)
        if nterms <= 0:
            raise ValueError("nterms must be positive")
        t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
        _check_observation_count(t_arr.size, nterms)
        delta_t = np.max(t_arr) - np.min(t_arr)
        Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
        frequency = f0 + df * np.arange(Nf, dtype=np.float64)
        result = tls.power(
            Nf,
            df,
            f0,
            t_arr,
            y_arr,
            dy_arr,
            backend=backend,
            solver=solver,
            normalization=normalization,
            nterms=nterms,
            autonan=autonan,
        )
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


class tlsdd:
    """Double-double FastChi2 periodogram wrapper returning ``mpmath`` values."""

    @staticmethod
    def power(
        Nf,
        df,
        f0,
        t,
        y,
        dy=None,
        backend="pswf43",
        solver="levinson",
        nterms=3,
        normalization="standard",
        *,
        autonan=True,
    ):
        """Return high-precision periodogram power."""
        mp = _require_mpmath()
        normalization = _check_normalization(normalization)
        Nf = int(Nf)
        nterms = int(nterms)
        df_mpf = mp.mpf(df)
        f0_mpf = mp.mpf(f0)
        if Nf <= 0:
            raise ValueError("Nf must be positive")
        if df_mpf <= 0:
            raise ValueError("df must be positive")
        if f0_mpf < 0:
            raise ValueError("f0 must be non-negative")
        if nterms <= 0:
            raise ValueError("nterms must be positive")

        backend = _backend_id(backend)
        solver = _solver_id(solver)
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
        cond = None if autonan else (DD * Nf)()
        cond_ptr = ctypes.POINTER(DD)() if cond is None else cond

        status = _load_library().tlsdd_fastchi2(
            t_dd,
            y_dd,
            dy_dd,
            len(t_mpf),
            float(f0_mpf),
            float(df_mpf),
            Nf,
            nterms,
            backend,
            solver,
            out,
            cond_ptr,
        )
        _raise_status("tlsdd_fastchi2", status)
        power = [_dd_to_mpf(out[i]) for i in range(Nf)]
        if normalization != "standard":
            chi2_ref = _mp_chi2_ref(y_mpf, dy_mpf)
            power = _normalize_mpf_power(
                power, normalization, len(t_mpf), nterms, chi2_ref
            )
        if autonan:
            return power
        condition = [_dd_to_mpf(cond[i]) for i in range(Nf)]
        return power, condition

    @staticmethod
    def autopower(
        x,
        y,
        dy=None,
        fmin=None,
        fmax=12,
        oversampling=5,
        normalization="standard",
        nterms=3,
        backend="pswf43",
        solver="levinson",
        *,
        autonan=True,
    ):
        """Return automatic high-precision frequency and power lists."""
        mp = _require_mpmath()
        nterms = int(nterms)
        if nterms <= 0:
            raise ValueError("nterms must be positive")
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
        result = tlsdd.power(
            Nf,
            df_mpf,
            f0_mpf,
            t_mpf,
            y_mpf,
            dy_mpf,
            backend=backend,
            solver=solver,
            normalization=normalization,
            nterms=nterms,
            autonan=autonan,
        )
        if autonan:
            return frequency, result
        power, cond = result
        return frequency, power, cond


__all__ = ["DD", "tlsf", "tls", "tlsdd", "trim_mantissa"]
