"""Utility helpers for inspecting periodogram peaks."""

from __future__ import annotations

import math
import operator
from collections.abc import MutableSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

try:
    from mpmath import mp as _mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    _mp = None


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


def get_peaks(freq, power, cond=None, threshold=0, *, x=None, nterms=None) -> Peaks:
    """Return strict local maxima refined by quadratic interpolation.

    A sample at index ``i`` is considered a peak when all three neighboring
    power samples are finite, ``power[i] > power[i - 1]``,
    ``power[i] > power[i + 1]``, and ``power[i] > threshold``. NaN samples
    are ignored for peak detection. The returned peaks are sorted by
    interpolated power in descending order.
    """

    freq_decimals = _infer_freq_decimals(freq, x=x, nterms=nterms)
    if _can_use_numpy_fast_path(freq, power, cond):
        return _get_peaks_numpy(
            freq, power, cond, threshold=threshold, freq_decimals=freq_decimals
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

    peaks = []
    for idx in range(1, len(power_values) - 1):
        power_window = power_values[idx - 1 : idx + 2]
        if not _all_finite(power_window):
            continue
        if not (
            power_values[idx] > power_values[idx - 1]
            and power_values[idx] > power_values[idx + 1]
            and power_values[idx] > threshold
        ):
            continue

        vertex_freq, vertex_power = _quadratic_vertex(
            freq_values[idx - 1],
            power_values[idx - 1],
            freq_values[idx],
            power_values[idx],
            freq_values[idx + 1],
            power_values[idx + 1],
        )
        if cond_values is None:
            peaks.append(Peak(vertex_freq, vertex_power))
        else:
            cond_window = cond_values[idx - 1 : idx + 2]
            if _all_finite(cond_window):
                vertex_cond = _quadratic_value(
                    vertex_freq,
                    freq_values[idx - 1],
                    cond_values[idx - 1],
                    freq_values[idx],
                    cond_values[idx],
                    freq_values[idx + 1],
                    cond_values[idx + 1],
                )
            else:
                vertex_cond = math.nan
            peaks.append(Peak(vertex_freq, vertex_power, vertex_cond))

    peaks.sort(key=lambda peak: peak.power, reverse=True)
    return Peaks(
        peaks, has_cond=cond_values is not None, freq_decimals=freq_decimals
    )


def _get_peaks_numpy(freq, power, cond, *, threshold, freq_decimals: int) -> Peaks:
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
    if freq_arr.size < 3:
        return Peaks(
            (),
            has_cond=cond_arr is not None,
            freq_decimals=freq_decimals,
        )

    p0 = power_arr[:-2]
    p1 = power_arr[1:-1]
    p2 = power_arr[2:]
    mask = (
        np.isfinite(p0)
        & np.isfinite(p1)
        & np.isfinite(p2)
        & (p1 > p0)
        & (p1 > p2)
        & (p1 > threshold)
    )
    idx = np.nonzero(mask)[0] + 1
    if idx.size == 0:
        return Peaks(
            (),
            has_cond=cond_arr is not None,
            freq_decimals=freq_decimals,
        )

    x0 = freq_arr[idx - 1]
    x1 = freq_arr[idx]
    x2 = freq_arr[idx + 1]
    y0 = power_arr[idx - 1]
    y1 = power_arr[idx]
    y2 = power_arr[idx + 1]
    vertex_freq, vertex_power = _quadratic_vertex_numpy(x0, y0, x1, y1, x2, y2)

    if cond_arr is None:
        order = np.argsort(vertex_power)[::-1]
        peaks = [Peak(vertex_freq[i], vertex_power[i]) for i in order]
    else:
        vertex_cond = np.full(vertex_power.shape, np.nan, dtype=np.float64)
        cond_finite = (
            np.isfinite(cond_arr[idx - 1])
            & np.isfinite(cond_arr[idx])
            & np.isfinite(cond_arr[idx + 1])
        )
        if np.any(cond_finite):
            vertex_cond[cond_finite] = _quadratic_value_numpy(
                vertex_freq[cond_finite],
                x0[cond_finite],
                cond_arr[idx[cond_finite] - 1],
                x1[cond_finite],
                cond_arr[idx[cond_finite]],
                x2[cond_finite],
                cond_arr[idx[cond_finite] + 1],
            )
        order = np.argsort(vertex_power)[::-1]
        peaks = [
            Peak(vertex_freq[i], vertex_power[i], vertex_cond[i]) for i in order
        ]
    return Peaks(peaks, has_cond=cond_arr is not None, freq_decimals=freq_decimals)


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
