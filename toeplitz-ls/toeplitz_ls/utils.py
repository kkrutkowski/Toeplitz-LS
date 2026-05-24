"""Utility helpers for inspecting periodogram peaks."""

from __future__ import annotations

import math
import operator
import ctypes
from collections.abc import MutableSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from ._fastchi2 import DD, _dd_to_mpf, _load_library

try:
    from mpmath import mp as _mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    _mp = None


_C_INT_MAX = 2**31 - 1
_UTILS_ARGTYPES_CONFIGURED = False
_UTILS_STATUS_MESSAGES = {
    -1: "invalid argument",
    -3: "allocation failure",
}


@dataclass(eq=False)
class Peak:
    """Interpolated periodogram peak."""

    freq: object
    power: object
    cond: object = math.nan

    def __eq__(self, other) -> bool:
        if not isinstance(other, Peak):
            return NotImplemented
        return (
            _values_equal(self.freq, other.freq)
            and _values_equal(self.power, other.power)
            and _values_equal(self.cond, other.cond)
        )


class Peaks(MutableSequence):
    """Mutable list-like collection of sorted periodogram peaks."""

    def __init__(
        self,
        peaks: Iterable[Peak] = (),
        *,
        has_cond: bool = False,
        freq_decimals: int = 10,
    ):
        self._peaks = list(peaks)
        self.has_cond = has_cond
        self.freq_decimals = freq_decimals

    def __len__(self) -> int:
        return len(self._peaks)

    def __iter__(self) -> Iterator[Peak]:
        return iter(self._peaks)

    def __getitem__(self, index):
        return self._peaks[index]

    def __setitem__(self, index, value: Peak) -> None:
        self._peaks[index] = value

    def __delitem__(self, index) -> None:
        del self._peaks[index]

    def insert(self, index: int, value: Peak) -> None:
        self._peaks.insert(index, value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._peaks!r}, "
            f"has_cond={self.has_cond!r}, freq_decimals={self.freq_decimals!r})"
        )

    def sort(self, *, key=None, reverse: bool = False) -> None:
        self._peaks.sort(key=key, reverse=reverse)

    def to_list(self) -> List[Peak]:
        """Return a shallow list copy of the peaks."""

        return list(self._peaks)

    def print(
        self,
        n: Optional[int] = 10,
        *,
        precision: int = 3,
        freq_decimals: Optional[int] = None,
    ) -> None:
        """Print the first ``n`` peaks as a terminal table."""

        freq_decimals = self.freq_decimals if freq_decimals is None else freq_decimals
        freq_decimals = _check_nonnegative_int("freq_decimals", freq_decimals)
        precision = _check_nonnegative_int("precision", precision)
        headers = ["rank", "freq", "power"]
        if self.has_cond:
            headers.append("cond")

        rows = []
        for rank, peak in enumerate(_limited(self._peaks, n), start=1):
            row = [
                str(rank),
                _format_fixed(peak.freq, freq_decimals),
                _format_fixed(peak.power, precision),
            ]
            if self.has_cond:
                row.append(_format_fixed(peak.cond, precision))
            rows.append(row)

        widths = [
            max(len(headers[col]), *(len(row[col]) for row in rows))
            if rows
            else len(headers[col])
            for col in range(len(headers))
        ]
        print(_format_table_row(headers, widths))
        print(_format_table_row(["-" * width for width in widths], widths))
        for row in rows:
            print(_format_table_row(row, widths))

    def save(
        self,
        path,
        n: Optional[int] = None,
        *,
        precision: int = 3,
        freq_decimals: Optional[int] = None,
    ) -> None:
        """Write the first ``n`` peaks to a tab-separated file."""

        freq_decimals = self.freq_decimals if freq_decimals is None else freq_decimals
        freq_decimals = _check_nonnegative_int("freq_decimals", freq_decimals)
        precision = _check_nonnegative_int("precision", precision)
        headers = ["rank", "freq", "power"]
        if self.has_cond:
            headers.append("cond")

        lines = ["\t".join(headers)]
        for rank, peak in enumerate(_limited(self._peaks, n), start=1):
            row = [
                str(rank),
                _format_fixed(peak.freq, freq_decimals),
                _format_fixed(peak.power, precision),
            ]
            if self.has_cond:
                row.append(_format_fixed(peak.cond, precision))
            lines.append("\t".join(row))

        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_utils_library():
    global _UTILS_ARGTYPES_CONFIGURED
    lib = _load_library()
    if _UTILS_ARGTYPES_CONFIGURED:
        return lib

    lib.tlsf_get_peaks.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tlsf_get_peaks.restype = ctypes.c_int

    lib.tls_get_peaks.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tls_get_peaks.restype = ctypes.c_int

    lib.tlsdd_get_peaks.argtypes = [
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.c_int,
        ctypes.c_int,
        DD,
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.POINTER(DD),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tlsdd_get_peaks.restype = ctypes.c_int

    _UTILS_ARGTYPES_CONFIGURED = True
    return lib


def _raise_utils_status(name: str, status: int) -> None:
    if status == -1:
        raise ValueError(f"{name} failed: invalid argument")
    if status == -3:
        raise MemoryError(f"{name} failed: allocation failure")
    if status < 0:
        message = _UTILS_STATUS_MESSAGES.get(status, f"status {status}")
        raise RuntimeError(f"{name} failed: {message}")


def _check_c_length(name: str, n: int) -> int:
    if n > _C_INT_MAX:
        raise ValueError(f"{name} length must be at most {_C_INT_MAX}")
    return int(n)


def _peak_capacity(num_samples: int, num_peaks) -> int:
    max_possible = max(0, num_samples - 2)
    if num_peaks is None:
        return _check_c_length("num_peaks", max_possible)
    requested = _check_nonnegative_int("num_peaks", num_peaks)
    if requested > _C_INT_MAX:
        raise ValueError(f"num_peaks must be at most {_C_INT_MAX}")
    return min(requested, max_possible)


def get_peaks(
    freq, power, cond=None, threshold=0, num_peaks=25, *, x=None, nterms=None
) -> Peaks:
    """Return strict local maxima refined by quadratic interpolation.

    A sample at index ``i`` is considered a peak when all three neighboring
    power samples are finite, ``power[i] > power[i - 1]``,
    ``power[i] > power[i + 1]``, and ``power[i] > threshold``. NaN samples
    are ignored for peak detection. The returned peaks are sorted by
    interpolated power in descending order. By default only the strongest
    25 peaks are returned; pass ``num_peaks=None`` to return all peaks.
    """

    freq_decimals = _infer_freq_decimals(freq, x=x, nterms=nterms)
    if _can_use_numpy_fast_path(freq, power, cond):
        return _get_peaks_numpy_native(
            freq,
            power,
            cond,
            threshold=threshold,
            num_peaks=num_peaks,
            freq_decimals=freq_decimals,
        )

    freq_values = _as_sequence(freq, "freq")
    power_values = _as_sequence(power, "power")
    cond_values = None if cond is None else _as_sequence(cond, "cond")
    threshold = _check_finite_scalar(threshold, "threshold")

    _check_lengths(freq_values, power_values, cond_values)
    _check_finite_sequence(freq_values, "freq")
    _check_numeric_sequence(power_values, "power")
    if cond_values is not None:
        _check_numeric_sequence(cond_values, "cond")
    _check_increasing(freq_values)

    if _should_use_mpmath_peak_path(freq_values, power_values, cond_values, threshold):
        return _get_peaks_mpmath_native(
            freq_values,
            power_values,
            cond_values,
            threshold=threshold,
            num_peaks=num_peaks,
            freq_decimals=freq_decimals,
        )

    return _get_peaks_double_native(
        freq_values,
        power_values,
        cond_values,
        threshold=threshold,
        num_peaks=num_peaks,
        freq_decimals=freq_decimals,
    )


def _get_peaks_numpy_native(
    freq, power, cond, *, threshold, num_peaks, freq_decimals: int
) -> Peaks:
    freq_arr = np.asarray(freq)
    power_arr = np.asarray(power)
    cond_arr = None if cond is None else np.asarray(cond)
    threshold = float(_check_finite_scalar(threshold, "threshold"))

    if freq_arr.ndim != 1:
        raise ValueError("freq must be one-dimensional")
    if power_arr.ndim != 1:
        raise ValueError("power must be one-dimensional")
    if cond_arr is not None and cond_arr.ndim != 1:
        raise ValueError("cond must be one-dimensional")
    if freq_arr.size != power_arr.size:
        raise ValueError("freq and power must have the same length")
    if cond_arr is not None and freq_arr.size != cond_arr.size:
        raise ValueError("freq, power, and cond must have the same length")

    if not np.all(np.isfinite(freq_arr)):
        raise ValueError("freq entries must be finite")
    if np.any(freq_arr[1:] <= freq_arr[:-1]):
        raise ValueError("freq must be strictly increasing")

    float_threshold_ok = abs(threshold) <= float(np.finfo(np.float32).max)
    use_float = (
        freq_arr.dtype == np.dtype(np.float32)
        and power_arr.dtype == np.dtype(np.float32)
        and (cond_arr is None or cond_arr.dtype == np.dtype(np.float32))
        and float_threshold_ok
    )
    dtype = np.float32 if use_float else np.float64
    freq_arr = np.ascontiguousarray(freq_arr, dtype=dtype)
    power_arr = np.ascontiguousarray(power_arr, dtype=dtype)
    if cond_arr is not None:
        cond_arr = np.ascontiguousarray(cond_arr, dtype=dtype)

    if dtype == np.float32:
        return _call_tlsf_get_peaks(
            freq_arr,
            power_arr,
            cond_arr,
            max_peaks=_peak_capacity(freq_arr.size, num_peaks),
            threshold=np.float32(threshold),
            freq_decimals=freq_decimals,
        )
    return _call_tls_get_peaks(
        freq_arr,
        power_arr,
        cond_arr,
        max_peaks=_peak_capacity(freq_arr.size, num_peaks),
        threshold=threshold,
        freq_decimals=freq_decimals,
    )


def _get_peaks_double_native(
    freq_values, power_values, cond_values, *, threshold, num_peaks, freq_decimals: int
) -> Peaks:
    freq_arr = np.ascontiguousarray(freq_values, dtype=np.float64)
    power_arr = np.ascontiguousarray(power_values, dtype=np.float64)
    cond_arr = (
        None
        if cond_values is None
        else np.ascontiguousarray(cond_values, dtype=np.float64)
    )
    return _call_tls_get_peaks(
        freq_arr,
        power_arr,
        cond_arr,
        max_peaks=_peak_capacity(freq_arr.size, num_peaks),
        threshold=float(threshold),
        freq_decimals=freq_decimals,
    )


def _call_tlsf_get_peaks(
    freq_arr, power_arr, cond_arr, *, max_peaks: int, threshold, freq_decimals: int
) -> Peaks:
    n_input = _check_c_length("freq", freq_arr.size)
    out_freq = np.empty(max_peaks, dtype=np.float32)
    out_power = np.empty(max_peaks, dtype=np.float32)
    out_cond = None if cond_arr is None else np.empty(max_peaks, dtype=np.float32)
    count = ctypes.c_int()
    cond_ptr = (
        ctypes.POINTER(ctypes.c_float)()
        if cond_arr is None
        else cond_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    out_cond_ptr = (
        ctypes.POINTER(ctypes.c_float)()
        if out_cond is None
        else out_cond.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    status = _load_utils_library().tlsf_get_peaks(
        freq_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        power_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cond_ptr,
        n_input,
        max_peaks,
        threshold,
        out_freq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_power.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_cond_ptr,
        ctypes.byref(count),
    )
    _raise_utils_status("tlsf_get_peaks", status)

    n = count.value
    if out_cond is None:
        peaks = [Peak(out_freq[i], out_power[i]) for i in range(n)]
    else:
        peaks = [Peak(out_freq[i], out_power[i], out_cond[i]) for i in range(n)]
    return Peaks(peaks, has_cond=out_cond is not None, freq_decimals=freq_decimals)


def _call_tls_get_peaks(
    freq_arr, power_arr, cond_arr, *, max_peaks: int, threshold, freq_decimals: int
) -> Peaks:
    n_input = _check_c_length("freq", freq_arr.size)
    out_freq = np.empty(max_peaks, dtype=np.float64)
    out_power = np.empty(max_peaks, dtype=np.float64)
    out_cond = None if cond_arr is None else np.empty(max_peaks, dtype=np.float64)
    count = ctypes.c_int()
    cond_ptr = (
        ctypes.POINTER(ctypes.c_double)()
        if cond_arr is None
        else cond_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    out_cond_ptr = (
        ctypes.POINTER(ctypes.c_double)()
        if out_cond is None
        else out_cond.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    status = _load_utils_library().tls_get_peaks(
        freq_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        power_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cond_ptr,
        n_input,
        max_peaks,
        threshold,
        out_freq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_power.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_cond_ptr,
        ctypes.byref(count),
    )
    _raise_utils_status("tls_get_peaks", status)

    n = count.value
    if out_cond is None:
        peaks = [Peak(out_freq[i], out_power[i]) for i in range(n)]
    else:
        peaks = [Peak(out_freq[i], out_power[i], out_cond[i]) for i in range(n)]
    return Peaks(peaks, has_cond=out_cond is not None, freq_decimals=freq_decimals)


def _should_use_mpmath_peak_path(freq_values, power_values, cond_values, threshold) -> bool:
    if _mp is None:
        return False
    if isinstance(threshold, _mp.mpf):
        return True
    if any(isinstance(value, _mp.mpf) for value in freq_values):
        return True
    if any(isinstance(value, _mp.mpf) for value in power_values):
        return True
    return cond_values is not None and any(isinstance(value, _mp.mpf) for value in cond_values)


def _get_peaks_mpmath_native(
    freq_values, power_values, cond_values, *, threshold, num_peaks, freq_decimals: int
) -> Peaks:
    mp = _mp
    freq_mpf = _mpf_values(freq_values)
    power_mpf = _mpf_values(power_values)
    cond_mpf = None if cond_values is None else _mpf_values(cond_values)
    threshold_dd = _mpf_to_dd_value(mp.mpf(threshold))

    n = _check_c_length("freq", len(freq_mpf))
    max_peaks = _peak_capacity(n, num_peaks)
    freq_dd = (DD * n)(*(_mpf_to_dd_value(value) for value in freq_mpf))
    power_dd = (DD * n)(*(_mpf_to_dd_value(value) for value in power_mpf))
    cond_dd = None if cond_mpf is None else (DD * n)(*(_mpf_to_dd_value(value) for value in cond_mpf))
    out_freq = (DD * max_peaks)()
    out_power = (DD * max_peaks)()
    out_cond = None if cond_dd is None else (DD * max_peaks)()
    count = ctypes.c_int()
    cond_ptr = ctypes.POINTER(DD)() if cond_dd is None else cond_dd
    out_cond_ptr = ctypes.POINTER(DD)() if out_cond is None else out_cond

    status = _load_utils_library().tlsdd_get_peaks(
        freq_dd,
        power_dd,
        cond_ptr,
        n,
        max_peaks,
        threshold_dd,
        out_freq,
        out_power,
        out_cond_ptr,
        ctypes.byref(count),
    )
    _raise_utils_status("tlsdd_get_peaks", status)

    n_peaks = count.value
    if out_cond is None:
        peaks = [Peak(_dd_to_mpf(out_freq[i]), _dd_to_mpf(out_power[i])) for i in range(n_peaks)]
    else:
        peaks = [
            Peak(_dd_to_mpf(out_freq[i]), _dd_to_mpf(out_power[i]), _dd_to_mpf(out_cond[i]))
            for i in range(n_peaks)
        ]
    return Peaks(peaks, has_cond=out_cond is not None, freq_decimals=freq_decimals)


def _mpf_to_dd_value(value):
    mp = _mp
    value = value if isinstance(value, mp.mpf) else mp.mpf(value)
    hi = float(value)
    lo = float(value - mp.mpf(hi))
    return DD(hi, lo)


def _mpf_values(values):
    mp = _mp
    if all(isinstance(value, mp.mpf) for value in values):
        return list(values)
    return [mp.mpf(value) for value in values]


def _can_use_numpy_fast_path(freq, power, cond) -> bool:
    if not (isinstance(freq, np.ndarray) and isinstance(power, np.ndarray)):
        return False
    if not (_is_fast_float_dtype(freq.dtype) and _is_fast_float_dtype(power.dtype)):
        return False
    return cond is None or (
        isinstance(cond, np.ndarray) and _is_fast_float_dtype(cond.dtype)
    )


def _is_fast_float_dtype(dtype) -> bool:
    dtype = np.dtype(dtype)
    return dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64)


def _as_sequence(values, name: str) -> List[object]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return list(values)

    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional iterable") from exc
    return result


def _check_lengths(
    freq: Sequence[object], power: Sequence[object], cond: Optional[Sequence[object]]
) -> None:
    if len(freq) != len(power):
        raise ValueError("freq and power must have the same length")
    if cond is not None and len(freq) != len(cond):
        raise ValueError("freq, power, and cond must have the same length")


def _check_finite_sequence(values: Sequence[object], name: str) -> None:
    for value in values:
        if not _is_finite(value):
            raise ValueError(f"{name} entries must be finite")


def _check_numeric_sequence(values: Sequence[object], name: str) -> None:
    for value in values:
        if not _is_numeric(value):
            raise ValueError(f"{name} entries must be numeric")


def _check_finite_scalar(value, name: str):
    if not _is_finite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _check_positive_int(name: str, value) -> int:
    result = _check_nonnegative_int(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _check_nonnegative_int(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _check_increasing(values: Sequence[object]) -> None:
    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            raise ValueError("freq must be strictly increasing")


def _is_finite(value) -> bool:
    if _mp is not None and isinstance(value, _mp.mpf):
        return bool(_mp.isfinite(value))
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _is_numeric(value) -> bool:
    if _mp is not None and isinstance(value, _mp.mpf):
        return True
    try:
        float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _all_finite(values: Sequence[object]) -> bool:
    return all(_is_finite(value) for value in values)


def _values_equal(left, right) -> bool:
    if _is_nan(left) and _is_nan(right):
        return True
    try:
        return bool(left == right)
    except ValueError:
        return False


def _is_nan(value) -> bool:
    if _mp is not None and isinstance(value, _mp.mpf):
        return bool(_mp.isnan(value))
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _quadratic_vertex(x0, y0, x1, y1, x2, y2):
    slope01, curvature = _quadratic_terms(x0, y0, x1, y1, x2, y2)
    if curvature == 0:
        return x1, y1
    linear = slope01 - curvature * (x0 + x1)
    vertex_x = -linear / (2 * curvature)
    vertex_y = _evaluate_quadratic(vertex_x, x0, y0, x1, slope01, curvature)
    return vertex_x, vertex_y


def _quadratic_vertex_numpy(x0, y0, x1, y1, x2, y2):
    slope01, curvature = _quadratic_terms_numpy(x0, y0, x1, y1, x2, y2)
    linear = slope01 - curvature * (x0 + x1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex_x = np.where(curvature != 0, -linear / (2 * curvature), x1)
    vertex_y = _evaluate_quadratic(vertex_x, x0, y0, x1, slope01, curvature)
    return vertex_x, vertex_y


def _quadratic_value(x, x0, y0, x1, y1, x2, y2):
    slope01, curvature = _quadratic_terms(x0, y0, x1, y1, x2, y2)
    return _evaluate_quadratic(x, x0, y0, x1, slope01, curvature)


def _quadratic_value_numpy(x, x0, y0, x1, y1, x2, y2):
    slope01, curvature = _quadratic_terms_numpy(x0, y0, x1, y1, x2, y2)
    return _evaluate_quadratic(x, x0, y0, x1, slope01, curvature)


def _quadratic_terms(x0, y0, x1, y1, x2, y2):
    slope01 = (y1 - y0) / (x1 - x0)
    slope12 = (y2 - y1) / (x2 - x1)
    curvature = (slope12 - slope01) / (x2 - x0)
    return slope01, curvature


def _quadratic_terms_numpy(x0, y0, x1, y1, x2, y2):
    with np.errstate(divide="ignore", invalid="ignore"):
        slope01 = (y1 - y0) / (x1 - x0)
        slope12 = (y2 - y1) / (x2 - x1)
        curvature = (slope12 - slope01) / (x2 - x0)
    return slope01, curvature


def _evaluate_quadratic(x, x0, y0, x1, slope01, curvature):
    return y0 + slope01 * (x - x0) + curvature * (x - x0) * (x - x1)


def _infer_freq_decimals(freq, *, x, nterms) -> int:
    if x is not None and nterms is not None:
        return _freq_decimals_from_timespan(x, nterms)
    return _freq_decimals_from_grid(freq)


def _freq_decimals_from_timespan(x, nterms) -> int:
    nterms = _check_positive_int("nterms", nterms)
    x_values = _as_sequence(x, "x")
    if not x_values:
        raise ValueError("x must not be empty")
    _check_finite_sequence(x_values, "x")
    span = max(x_values) - min(x_values)
    if span <= 0:
        raise ValueError("x must span a positive interval")
    return max(0, int(math.ceil(math.log10(float(span * nterms)) + 2)))


def _freq_decimals_from_grid(freq) -> int:
    try:
        if isinstance(freq, np.ndarray):
            if freq.ndim != 1 or freq.size < 2:
                return 10
            spacing = freq[1] - freq[0]
        else:
            if len(freq) < 2:
                return 10
            spacing = freq[1] - freq[0]
        spacing = abs(float(spacing))
        if not math.isfinite(spacing) or spacing <= 0:
            return 10
        return max(0, int(-math.floor(math.log10(0.05 * spacing))))
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return 10


def _limited(values: Sequence[Peak], n: Optional[int]) -> Sequence[Peak]:
    if n is None:
        return values
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise ValueError("n must be an integer") from exc
    if n < 0:
        raise ValueError("n must be non-negative")
    return values[:n]


def _format_fixed(value, decimals: int) -> str:
    try:
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    cells = " | ".join(value.rjust(width) for value, width in zip(values, widths))
    return f"| {cells} |"


__all__ = ["Peak", "Peaks", "get_peaks"]
