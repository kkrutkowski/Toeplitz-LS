"""Uniform periodogram adapters used by the dataset comparison harness."""

from __future__ import annotations

import ctypes
import math
import operator
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOEPLITZ_PYTHON = ROOT / "toeplitz-ls"
if str(TOEPLITZ_PYTHON) not in sys.path:
    sys.path.insert(0, str(TOEPLITZ_PYTHON))

from toeplitz_ls import tls as _tls  # noqa: E402
from toeplitz_ls import tlsf as _tlsf  # noqa: E402

from astropy_impl.chi2_impl import lombscargle_chi2  # noqa: E402
from astropy_impl.fast_impl import lombscargle_fast  # noqa: E402
from astropy_impl.fastchi2_impl import lombscargle_fastchi2  # noqa: E402
from astropy_impl.slow_impl import lombscargle_slow  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"
AOVDIST_LIBRARY = BUILD_DIR / "libaovdist.so"
FASTCHI2_LIBRARY = BUILD_DIR / "libfastchi2_103.so"
_C_INT_MAX = 2**31 - 1


class UnsupportedMethodError(ValueError):
    """Raised when an implementation cannot support the requested order."""


@dataclass(frozen=True)
class FrequencyGrid:
    """Regular frequency grid used for a periodogram calculation."""

    frequency: np.ndarray
    Nf: int
    df: float
    f0: float


def trim_mantissa(value: float, bits: int) -> float:
    """Return a value with a truncated binary mantissa."""
    mantissa, exponent = math.frexp(value)
    mantissa = math.floor(mantissa * (1 << bits)) / (1 << bits)
    return math.ldexp(mantissa, exponent)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive")
    if result > _C_INT_MAX:
        raise ValueError(f"{name} must be at most {_C_INT_MAX}")
    return result


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def make_frequency_grid(
    t,
    *,
    fmin: float | None = None,
    fmax: float = 4.0,
    oversampling: float = 5.0,
    nterms: int = 1,
) -> FrequencyGrid:
    """Construct the same regular automatic grid used by ``tls.autopower``."""
    nterms = _positive_int("nterms", nterms)
    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size == 0:
        raise ValueError("t must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("t entries must be finite")

    delta_t = float(np.max(t_arr) - np.min(t_arr))
    fmax = _finite_float("fmax", fmax)
    oversampling = _finite_float("oversampling", oversampling)
    if delta_t <= 0.0:
        raise ValueError("t must span a positive interval")
    if fmax <= 0.0:
        raise ValueError("fmax must be positive")
    if oversampling <= 0.0:
        raise ValueError("oversampling must be positive")

    df = trim_mantissa(1.0 / (oversampling * delta_t * nterms), 5)
    if fmin is None:
        first_multiple = math.floor((1.0 / delta_t) / df) + 1
    else:
        fmin = _finite_float("fmin", fmin)
        if fmin < 0.0:
            raise ValueError("fmin must be non-negative")
        first_multiple = math.ceil((fmin / delta_t) / df)

    f0 = first_multiple * df
    Nf = math.floor((fmax - f0) / df) + 1
    if Nf <= 0:
        raise ValueError("automatic frequency grid is empty")
    Nf = _positive_int("Nf", Nf)
    frequency = f0 + df * np.arange(Nf, dtype=np.float64)
    return FrequencyGrid(frequency=frequency, Nf=Nf, df=df, f0=f0)


def _check_power_request(Nf, df, f0, nterms, normalization):
    Nf = _positive_int("Nf", Nf)
    nterms = _positive_int("nterms", nterms)
    df = _finite_float("df", df)
    f0 = _finite_float("f0", f0)
    if df <= 0.0:
        raise ValueError("df must be positive")
    if f0 < 0.0:
        raise ValueError("f0 must be non-negative")
    if normalization != "standard":
        raise ValueError("comparison adapters support normalization='standard' only")
    return Nf, df, f0, nterms


def _as_frequency(Nf, df, f0, frequency):
    expected = f0 + df * np.arange(Nf, dtype=np.float64)
    if frequency is None:
        return expected
    result = np.ascontiguousarray(frequency, dtype=np.float64)
    if result.ndim != 1 or result.size != Nf:
        raise ValueError("frequency must be a one-dimensional array of length Nf")
    tolerance = max(abs(df) * 1e-10, np.finfo(np.float64).eps * 32)
    if not np.allclose(result, expected, rtol=0.0, atol=tolerance):
        raise ValueError("frequency does not match f0 + df * arange(Nf)")
    return result


def _inputs(t, y, dy, dtype=np.float64):
    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=dtype)
    dy_arr = None if dy is None else np.ascontiguousarray(dy, dtype=dtype)
    if t_arr.ndim != 1 or y_arr.ndim != 1 or (
        dy_arr is not None and dy_arr.ndim != 1
    ):
        raise ValueError("t, y, and dy must be one-dimensional")
    if t_arr.size != y_arr.size or (
        dy_arr is not None and t_arr.size != dy_arr.size
    ):
        raise ValueError("t, y, and dy must have the same length")
    if t_arr.size < 1:
        raise ValueError("t and y must not be empty")
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("t and y entries must be finite")
    if dy_arr is not None:
        if not np.all(np.isfinite(dy_arr)) or np.any(dy_arr <= 0.0):
            raise ValueError("dy entries must be finite and positive")
    return t_arr, y_arr, dy_arr


class Method:
    """Base adapter implementing automatic-grid delegation to ``power``."""

    name = ""
    max_nterms: int | None = None

    def supports(self, nterms: int) -> bool:
        return self.max_nterms is None or nterms <= self.max_nterms

    def autopower(
        self,
        t,
        y,
        dy=None,
        fmin=None,
        fmax=4,
        oversampling=5,
        nterms=1,
        normalization="standard",
    ):
        grid = make_frequency_grid(
            t, fmin=fmin, fmax=fmax, oversampling=oversampling, nterms=nterms
        )
        power = self.power(
            grid.Nf,
            grid.df,
            grid.f0,
            t,
            y,
            dy,
            nterms=nterms,
            normalization=normalization,
            frequency=grid.frequency,
        )
        return grid.frequency, power


class TLSMethod(Method):
    name = "tls"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        kwargs = {"backend": "pswf43", "nterms": nterms, "normalization": normalization}
        if nterms > 1:
            kwargs["solver"] = "levinson"
        return _tls.power(Nf, df, f0, t, y, dy, **kwargs)


class TLSFMethod(Method):
    name = "tlsf"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        kwargs = {"backend": "pswf43", "nterms": nterms, "normalization": normalization}
        if nterms > 1:
            kwargs["solver"] = "levinson"
        return _tlsf.power(Nf, df, f0, t, y, dy, **kwargs)


class AstropySlowMethod(Method):
    name = "astropy-slow"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        freq = _as_frequency(Nf, df, f0, frequency)
        if nterms == 1:
            return lombscargle_slow(t, y, dy, freq, normalization=normalization)
        return lombscargle_chi2(
            t, y, dy, freq, normalization=normalization, nterms=nterms
        )


class AstropyFastMethod(Method):
    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        if nterms == 1:
            return lombscargle_fast(
                t, y, dy, f0, df, Nf, normalization=normalization, algorithm=self.algorithm
            )
        return lombscargle_fastchi2(
            t,
            y,
            dy,
            f0,
            df,
            Nf,
            normalization=normalization,
            nterms=nterms,
            algorithm=self.algorithm,
        )


class _Spec(ctypes.Structure):
    _fields_ = [
        ("nfr", ctypes.c_long),
        ("fr0", ctypes.c_double),
        ("frs", ctypes.c_double),
        ("th", ctypes.POINTER(ctypes.c_float)),
    ]


_AOV_LIBRARY = None
_FASTCHI2_LIBRARY = None


def _missing_library(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"{path} does not exist. Build comparison libraries with "
        "'make -C examples/8_Benchmark native'."
    )


def _load_aov_library():
    global _AOV_LIBRARY
    if _AOV_LIBRARY is not None:
        return _AOV_LIBRARY
    if not AOVDIST_LIBRARY.exists():
        raise _missing_library(AOVDIST_LIBRARY)
    lib = ctypes.CDLL(str(AOVDIST_LIBRARY))
    lib.AovDrv.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        _Spec,
    ]
    lib.AovDrv.restype = ctypes.c_int
    _AOV_LIBRARY = lib
    return lib


def _load_fastchi2_library():
    global _FASTCHI2_LIBRARY
    if _FASTCHI2_LIBRARY is not None:
        return _FASTCHI2_LIBRARY
    if not FASTCHI2_LIBRARY.exists():
        raise _missing_library(FASTCHI2_LIBRARY)
    lib = ctypes.CDLL(str(FASTCHI2_LIBRARY))
    lib.fastChi.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_double,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.fastChi.restype = None
    _FASTCHI2_LIBRARY = lib
    return lib


class AoVDISTMethod(Method):
    name = "aovdist"
    _AMH = 3
    _AMHW = 7
    _SP_MODEL = 3

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        t_arr, y_arr, dy_arr = _inputs(t, y, dy, np.float32)
        out = np.empty(Nf, dtype=np.float32)
        values = y_arr.copy()
        float_ptr = ctypes.POINTER(ctypes.c_float)
        if dy_arr is None:
            method = self._AMH
            weights_ptr = float_ptr()
        else:
            method = self._AMHW
            weights = np.ascontiguousarray(1.0 / (dy_arr * dy_arr), dtype=np.float32)
            weights_ptr = weights.ctypes.data_as(float_ptr)
        spec = _Spec(Nf, f0, df, out.ctypes.data_as(float_ptr))
        status = _load_aov_library().AovDrv(
            method,
            self._SP_MODEL,
            t_arr.size,
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            values.ctypes.data_as(float_ptr),
            weights_ptr,
            2 * nterms + 1,
            1,
            spec,
        )
        if status != 0:
            raise RuntimeError(f"AovDrv failed with status {status}")
        return out


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


class FastChi2103Method(Method):
    name = "fastchi2_103"
    max_nterms = 15

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        if not self.supports(nterms):
            raise UnsupportedMethodError("fastchi2_103 supports nterms up to 15")
        t_arr, y_arr, dy_arr = _inputs(t, y, dy, np.float32)
        if dy_arr is None:
            dy_arr = np.ones(t_arr.shape, dtype=np.float32)

        grid_bin = f0 / df
        first_bin = int(round(grid_bin))
        if first_bin < 1 or not math.isclose(grid_bin, first_bin, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("fastchi2_103 requires f0 to be a positive integer multiple of df")

        nchivalues = first_bin + Nf
        nrealpoints = _next_power_of_two(max(2, 4 * nterms * nchivalues))
        deltat = 1.0 / (nrealpoints * df)
        tstart = float(np.min(t_arr) - deltat / 2.0)
        time_bins = ((t_arr - tstart) / deltat).astype(np.int64)
        if np.any(time_bins < 0) or np.any(time_bins >= nrealpoints):
            raise ValueError("fastchi2_103 grid does not cover the observation times")

        weights = 1.0 / np.square(dy_arr.astype(np.float64))
        centered = np.ascontiguousarray(
            y_arr.astype(np.float64) - np.average(y_arr.astype(np.float64), weights=weights),
            dtype=np.float32,
        )
        chi2_ref = float(np.sum(np.square(centered.astype(np.float64) / dy_arr)))
        if not math.isfinite(chi2_ref) or chi2_ref <= 0.0:
            raise ValueError("fastchi2_103 requires observations with nonzero variance")

        dovstorage = np.empty(nrealpoints, dtype=np.float32)
        oovstorage = np.empty(nrealpoints, dtype=np.float32)
        reduction = np.zeros(nchivalues, dtype=np.float32)
        float_ptr = ctypes.POINTER(ctypes.c_float)
        _load_fastchi2_library().fastChi(
            t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            centered.ctypes.data_as(float_ptr),
            dy_arr.ctypes.data_as(float_ptr),
            t_arr.size,
            nterms,
            deltat,
            nrealpoints,
            nchivalues,
            tstart,
            dovstorage.ctypes.data_as(float_ptr),
            oovstorage.ctypes.data_as(float_ptr),
            reduction.ctypes.data_as(float_ptr),
        )
        return reduction[first_bin:nchivalues] / chi2_ref


class NiftyMethod(Method):
    name = "nifty"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        Nf, df, f0, nterms = _check_power_request(Nf, df, f0, nterms, normalization)
        import nifty_ls

        backend = "finufft" if nterms == 1 else "finufft_chi2"
        fmax = f0 + df * (Nf - 1)
        result = nifty_ls.lombscargle(
            np.ascontiguousarray(t, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            dy=None if dy is None else np.ascontiguousarray(dy, dtype=np.float64),
            fmin=f0,
            fmax=fmax,
            Nf=Nf,
            center_data=True,
            fit_mean=True,
            normalization=normalization,
            assume_sorted_t=True,
            nterms=nterms,
            backend=backend,
            nthreads=1,
        )
        if result.Nf != Nf or not math.isclose(result.fmin, f0) or not math.isclose(result.df, df):
            raise RuntimeError("Nifty returned a frequency grid different from the requested grid")
        return np.asarray(result.power)


METHODS = {
    method.name: method
    for method in (
        TLSMethod(),
        TLSFMethod(),
        FastChi2103Method(),
        AoVDISTMethod(),
        AstropySlowMethod(),
        AstropyFastMethod("astropy-lra", "lra"),
        AstropyFastMethod("astropy-fasper", "fasper"),
        NiftyMethod(),
    )
}

__all__ = ["FrequencyGrid", "METHODS", "Method", "UnsupportedMethodError", "make_frequency_grid"]
