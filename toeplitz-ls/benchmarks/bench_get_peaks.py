from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from toeplitz_ls import Peak, get_peaks  # noqa: E402

try:
    from mpmath import mp
except ImportError:  # pragma: no cover
    mp = None


def _quadratic_terms(x0, y0, x1, y1, x2, y2):
    slope01 = (y1 - y0) / (x1 - x0)
    slope12 = (y2 - y1) / (x2 - x1)
    curvature = (slope12 - slope01) / (x2 - x0)
    return slope01, curvature


def _evaluate_quadratic(x, x0, y0, x1, slope01, curvature):
    return y0 + slope01 * (x - x0) + curvature * (x - x0) * (x - x1)


def old_numpy_get_peaks(freq, power, cond=None, threshold=0, num_peaks=25):
    freq_arr = np.asarray(freq)
    power_arr = np.asarray(power)
    cond_arr = None if cond is None else np.asarray(cond)

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
        return []

    x0 = freq_arr[idx - 1]
    x1 = freq_arr[idx]
    x2 = freq_arr[idx + 1]
    y0 = power_arr[idx - 1]
    y1 = power_arr[idx]
    y2 = power_arr[idx + 1]
    slope01, curvature = _quadratic_terms(x0, y0, x1, y1, x2, y2)
    linear = slope01 - curvature * (x0 + x1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex_freq = np.where(curvature != 0, -linear / (2 * curvature), x1)
    vertex_power = _evaluate_quadratic(vertex_freq, x0, y0, x1, slope01, curvature)
    order = np.argsort(vertex_power)[::-1]

    if cond_arr is None:
        peaks = [Peak(vertex_freq[i], vertex_power[i]) for i in order]
        return peaks if num_peaks is None else peaks[:num_peaks]

    vertex_cond = np.full(vertex_power.shape, np.nan, dtype=np.float64)
    cond_finite = (
        np.isfinite(cond_arr[idx - 1])
        & np.isfinite(cond_arr[idx])
        & np.isfinite(cond_arr[idx + 1])
    )
    if np.any(cond_finite):
        c0 = cond_arr[idx[cond_finite] - 1]
        c1 = cond_arr[idx[cond_finite]]
        c2 = cond_arr[idx[cond_finite] + 1]
        cx0 = x0[cond_finite]
        cx1 = x1[cond_finite]
        cx2 = x2[cond_finite]
        cslope01, ccurvature = _quadratic_terms(cx0, c0, cx1, c1, cx2, c2)
        vertex_cond[cond_finite] = _evaluate_quadratic(
            vertex_freq[cond_finite], cx0, c0, cx1, cslope01, ccurvature
        )
    peaks = [Peak(vertex_freq[i], vertex_power[i], vertex_cond[i]) for i in order]
    return peaks if num_peaks is None else peaks[:num_peaks]


def old_mpmath_get_peaks(freq, power, cond=None, threshold=0, num_peaks=25):
    peaks = []
    for idx in range(1, len(power) - 1):
        if not all(mp.isfinite(value) for value in power[idx - 1 : idx + 2]):
            continue
        if not (
            power[idx] > power[idx - 1]
            and power[idx] > power[idx + 1]
            and power[idx] > threshold
        ):
            continue

        x0, x1, x2 = freq[idx - 1], freq[idx], freq[idx + 1]
        y0, y1, y2 = power[idx - 1], power[idx], power[idx + 1]
        slope01, curvature = _quadratic_terms(x0, y0, x1, y1, x2, y2)
        if curvature == 0:
            vertex_freq = x1
            vertex_power = y1
        else:
            linear = slope01 - curvature * (x0 + x1)
            vertex_freq = -linear / (2 * curvature)
            vertex_power = _evaluate_quadratic(
                vertex_freq, x0, y0, x1, slope01, curvature
            )
        if cond is None:
            peaks.append(Peak(vertex_freq, vertex_power))
        elif all(mp.isfinite(value) for value in cond[idx - 1 : idx + 2]):
            c0, c1, c2 = cond[idx - 1], cond[idx], cond[idx + 1]
            cslope01, ccurvature = _quadratic_terms(x0, c0, x1, c1, x2, c2)
            peaks.append(
                Peak(
                    vertex_freq,
                    vertex_power,
                    _evaluate_quadratic(vertex_freq, x0, c0, x1, cslope01, ccurvature),
                )
            )
        else:
            peaks.append(Peak(vertex_freq, vertex_power, mp.nan))
    peaks.sort(key=lambda peak: peak.power, reverse=True)
    return peaks if num_peaks is None else peaks[:num_peaks]


def median_seconds(fn, repeat=7):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def max_delta(left, right):
    result = 0.0
    for a, b in zip(left, right):
        for attr in ("freq", "power", "cond"):
            av = getattr(a, attr)
            bv = getattr(b, attr)
            if math.isnan(float(av)) and math.isnan(float(bv)):
                continue
            result = max(result, abs(float(av) - float(bv)))
    return result


def bench_numpy():
    rng = np.random.default_rng(12345)
    n = 1_000_000
    freq = np.linspace(0.0, 2000.0, n, dtype=np.float64)
    power = (
        np.sin(freq * 0.91)
        + 0.3 * np.sin(freq * 7.3)
        + 0.03 * rng.standard_normal(n)
    )
    cond = 1.0 + 0.001 * freq + 0.000001 * freq * freq
    power[12345] = np.nan
    cond[54321] = np.nan

    old_time, old_result = median_seconds(
        lambda: old_numpy_get_peaks(freq, power, cond=cond, threshold=0.8), repeat=5
    )
    new_time, new_result = median_seconds(
        lambda: get_peaks(freq, power, cond=cond, threshold=0.8), repeat=5
    )
    print(
        "numpy/float64 "
        f"old={old_time:.6f}s new={new_time:.6f}s "
        f"speedup={old_time / new_time:.2f}x peaks={len(new_result)} "
        f"max_delta={max_delta(old_result, new_result):.3g}"
    )


def bench_mpmath():
    if mp is None:
        print("mpmath not installed; skipping tlsdd benchmark")
        return

    with mp.workprec(106):
        n = 4096
        freq = [mp.mpf(i) / 32 for i in range(n)]
        power = [mp.sin(mp.mpf("0.91") * x) + mp.mpf("0.3") * mp.sin(mp.mpf("7.3") * x) for x in freq]
        cond = [mp.mpf(1) + mp.mpf("0.001") * x + mp.mpf("0.000001") * x * x for x in freq]
        power[123] = mp.nan
        cond[456] = mp.nan
        threshold = mp.mpf("0.8")

        old_time, old_result = median_seconds(
            lambda: old_mpmath_get_peaks(freq, power, cond=cond, threshold=threshold),
            repeat=5,
        )
        new_time, new_result = median_seconds(
            lambda: get_peaks(freq, power, cond=cond, threshold=threshold), repeat=5
        )
        print(
            "mpmath/tlsdd "
            f"old={old_time:.6f}s new={new_time:.6f}s "
            f"speedup={old_time / new_time:.2f}x peaks={len(new_result)} "
            f"max_delta={max_delta(old_result, new_result):.3g}"
        )


def main():
    bench_numpy()
    bench_mpmath()


if __name__ == "__main__":
    main()
