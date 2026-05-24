from __future__ import annotations

import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from toeplitz_ls import Peak, Peaks, get_peaks  # noqa: E402


def _reference_get_peaks(freq, power, cond=None, threshold=0):
    peaks = []
    for idx in range(1, len(power) - 1):
        power_window = power[idx - 1 : idx + 2]
        if not all(_reference_is_finite(value) for value in power_window):
            continue
        if not (
            power[idx] > power[idx - 1]
            and power[idx] > power[idx + 1]
            and power[idx] > threshold
        ):
            continue

        vertex_freq, vertex_power = _reference_quadratic_vertex(
            freq[idx - 1],
            power[idx - 1],
            freq[idx],
            power[idx],
            freq[idx + 1],
            power[idx + 1],
        )
        if cond is None:
            peaks.append(Peak(vertex_freq, vertex_power))
        else:
            cond_window = cond[idx - 1 : idx + 2]
            if all(_reference_is_finite(value) for value in cond_window):
                vertex_cond = _reference_quadratic_value(
                    vertex_freq,
                    freq[idx - 1],
                    cond[idx - 1],
                    freq[idx],
                    cond[idx],
                    freq[idx + 1],
                    cond[idx + 1],
                )
            else:
                vertex_cond = math.nan if mp is None else mp.nan
            peaks.append(Peak(vertex_freq, vertex_power, vertex_cond))

    peaks.sort(key=lambda peak: peak.power, reverse=True)
    return peaks


def _reference_is_finite(value):
    if mp is not None and isinstance(value, mp.mpf):
        return bool(mp.isfinite(value))
    return math.isfinite(float(value))


def _reference_quadratic_vertex(x0, y0, x1, y1, x2, y2):
    slope01, curvature = _reference_quadratic_terms(x0, y0, x1, y1, x2, y2)
    if curvature == 0:
        return x1, y1
    linear = slope01 - curvature * (x0 + x1)
    vertex_x = -linear / (2 * curvature)
    vertex_y = _reference_evaluate_quadratic(
        vertex_x, x0, y0, x1, slope01, curvature
    )
    return vertex_x, vertex_y


def _reference_quadratic_value(x, x0, y0, x1, y1, x2, y2):
    slope01, curvature = _reference_quadratic_terms(x0, y0, x1, y1, x2, y2)
    return _reference_evaluate_quadratic(x, x0, y0, x1, slope01, curvature)


def _reference_quadratic_terms(x0, y0, x1, y1, x2, y2):
    slope01 = (y1 - y0) / (x1 - x0)
    slope12 = (y2 - y1) / (x2 - x1)
    curvature = (slope12 - slope01) / (x2 - x0)
    return slope01, curvature


def _reference_evaluate_quadratic(x, x0, y0, x1, slope01, curvature):
    return y0 + slope01 * (x - x0) + curvature * (x - x0) * (x - x1)


class PeakUtilityTests(unittest.TestCase):
    def test_quadratic_peak_is_refined_exactly(self):
        freq = [1.0, 2.0, 3.0, 4.0]
        power = [10.0 - (x - 2.25) ** 2 for x in freq]

        peaks = get_peaks(freq, power)

        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0].freq, 2.25)
        self.assertAlmostEqual(peaks[0].power, 10.0)
        self.assertTrue(math.isnan(peaks[0].cond))

    def test_condition_is_interpolated_at_power_vertex(self):
        freq = [1.0, 2.0, 3.0, 4.0]
        power = [10.0 - (x - 2.25) ** 2 for x in freq]
        cond = [5.0 + 2.0 * x + x * x for x in freq]

        peaks = get_peaks(freq, power, cond=cond)

        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0].freq, 2.25)
        self.assertAlmostEqual(peaks[0].power, 10.0)
        self.assertAlmostEqual(peaks[0].cond, 14.5625)

    def test_peaks_are_sorted_by_refined_power_descending(self):
        freq = list(range(7))
        power = [0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0]

        peaks = get_peaks(freq, power)

        self.assertEqual([peak.freq for peak in peaks], [1.0, 4.0])
        self.assertEqual([peak.power for peak in peaks], [3.0, 2.0])

    def test_threshold_uses_original_grid_sample(self):
        freq = [1.0, 2.0, 3.0, 4.0]
        power = [10.0 - (x - 2.25) ** 2 for x in freq]

        peaks = get_peaks(freq, power, threshold=9.95)

        self.assertEqual(len(peaks), 0)

    def test_endpoints_and_plateaus_are_ignored(self):
        self.assertEqual(len(get_peaks([0.0, 1.0, 2.0], [2.0, 1.0, 3.0])), 0)
        self.assertEqual(
            len(get_peaks([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 0.0])), 0
        )

    def test_nan_power_samples_are_accepted_and_skip_windows(self):
        freq = list(range(9))
        power = [0.0, 5.0, 0.0, math.nan, 3.0, 0.0, 0.0, 4.0, 0.0]

        peaks = get_peaks(freq, power)

        self.assertEqual([peak.freq for peak in peaks], [1.0, 7.0])
        self.assertEqual([peak.power for peak in peaks], [5.0, 4.0])

    def test_nan_condition_keeps_peak_with_nan_condition(self):
        peaks = get_peaks(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 0.0],
            cond=[2.0, math.nan, 4.0],
        )

        self.assertEqual(len(peaks), 1)
        self.assertTrue(peaks.has_cond)
        self.assertTrue(math.isnan(peaks[0].cond))

    def test_numpy_fast_path_matches_scalar_path(self):
        freq = np.linspace(0.0, 5.0, 11)
        power = np.array([0.0, 2.0, 0.0, math.nan, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        cond = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 2.0, math.nan, 5.0, 6.0, 7.0])

        fast = get_peaks(freq, power, cond=cond)
        scalar = _reference_get_peaks(list(freq), list(power), cond=list(cond))

        self.assertEqual(len(fast), len(scalar))
        for fast_peak, scalar_peak in zip(fast, scalar):
            self.assertAlmostEqual(float(fast_peak.freq), float(scalar_peak.freq))
            self.assertAlmostEqual(float(fast_peak.power), float(scalar_peak.power))
            if math.isnan(float(scalar_peak.cond)):
                self.assertTrue(math.isnan(float(fast_peak.cond)))
            else:
                self.assertAlmostEqual(float(fast_peak.cond), float(scalar_peak.cond))

    def test_native_numpy_matches_reference_for_float32_and_float64(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                freq = np.linspace(0.0, 25.0, 4097, dtype=dtype)
                power = (
                    np.sin(freq * dtype(1.7))
                    + dtype(0.15) * np.sin(freq * dtype(11.0))
                    + dtype(2.0)
                ).astype(dtype)
                power[111] = np.nan
                cond = (dtype(1.0) + freq * dtype(0.25) + freq * freq * dtype(0.01)).astype(dtype)
                cond[1000] = np.nan

                native = get_peaks(freq, power, cond=cond, threshold=1.9, num_peaks=None)
                reference = _reference_get_peaks(
                    list(freq), list(power), cond=list(cond), threshold=dtype(1.9)
                )

                self.assertEqual(len(native), len(reference))
                for native_peak, ref_peak in zip(native, reference):
                    places = 4 if dtype == np.float32 else 11
                    self.assertAlmostEqual(float(native_peak.freq), float(ref_peak.freq), places=places)
                    self.assertAlmostEqual(float(native_peak.power), float(ref_peak.power), places=places)
                    if math.isnan(float(ref_peak.cond)):
                        self.assertTrue(math.isnan(float(native_peak.cond)))
                    else:
                        self.assertAlmostEqual(float(native_peak.cond), float(ref_peak.cond), places=places)

    def test_python_sequences_use_native_double_and_match_reference(self):
        freq = [i * 0.05 for i in range(300)]
        power = [math.sin(3.2 * x) + 0.2 * math.cos(14.0 * x) for x in freq]
        power[44] = math.inf
        cond = [1.0 + x * x for x in freq]

        native = get_peaks(freq, power, cond=cond, threshold=0.7, num_peaks=None)
        reference = _reference_get_peaks(freq, power, cond=cond, threshold=0.7)

        self.assertEqual(len(native), len(reference))
        for native_peak, ref_peak in zip(native, reference):
            self.assertAlmostEqual(float(native_peak.freq), float(ref_peak.freq))
            self.assertAlmostEqual(float(native_peak.power), float(ref_peak.power))
            self.assertAlmostEqual(float(native_peak.cond), float(ref_peak.cond))

    def test_num_peaks_limits_returned_peaks(self):
        freq = list(range(101))
        power = [0.0] * len(freq)
        for idx in range(1, 100, 2):
            power[idx] = float(idx)

        default_limited = get_peaks(freq, power)
        explicit_limited = get_peaks(freq, power, num_peaks=7)
        unlimited = get_peaks(freq, power, num_peaks=None)

        self.assertEqual(len(default_limited), 25)
        self.assertEqual(len(explicit_limited), 7)
        self.assertEqual(len(unlimited), 50)
        self.assertEqual(
            [peak.power for peak in explicit_limited],
            [99.0, 97.0, 95.0, 93.0, 91.0, 89.0, 87.0],
        )

    def test_zero_num_peaks_returns_empty_collection(self):
        peaks = get_peaks([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], num_peaks=0)

        self.assertEqual(len(peaks), 0)

    def test_peaks_collection_is_mutable_and_list_like(self):
        peaks = Peaks([Peak(1.0, 2.0), Peak(3.0, 4.0)])

        peaks.append(Peak(5.0, 6.0))
        removed = peaks.pop(1)
        peaks[0] = Peak(7.0, 8.0)

        self.assertEqual(removed, Peak(3.0, 4.0))
        self.assertEqual(len(peaks), 2)
        self.assertEqual([peak.freq for peak in peaks], [7.0, 5.0])
        self.assertEqual(peaks.to_list(), [Peak(7.0, 8.0), Peak(5.0, 6.0)])

    def test_print_omits_or_includes_condition_column(self):
        no_cond = get_peaks([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        with_cond = get_peaks(
            [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], cond=[2.0, 3.0, 4.0]
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            no_cond.print()
        lines = buffer.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("|"))
        self.assertNotIn("cond", lines[0])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with_cond.print()
        self.assertIn("cond", buffer.getvalue().splitlines()[0])

    def test_print_defaults_to_top_ten_and_none_prints_all(self):
        peaks = Peaks([Peak(i, i) for i in range(12)], freq_decimals=0)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            peaks.print()
        self.assertEqual(len(buffer.getvalue().splitlines()), 12)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            peaks.print(None)
        self.assertEqual(len(buffer.getvalue().splitlines()), 14)

    def test_print_uses_pipe_table_and_default_fixed_decimals(self):
        peaks = get_peaks(
            np.array([0.0, 0.001, 0.002]),
            np.array([0.0, 1.0, 0.0]),
            cond=np.array([0.0, 2.0, 0.0]),
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            peaks.print()
        lines = buffer.getvalue().splitlines()

        self.assertEqual(peaks.freq_decimals, 5)
        self.assertIn("|", lines[0])
        self.assertIn("|    1 | 0.00100 | 1.000 | 2.000 |", lines[2])

    def test_optional_x_and_nterms_set_frequency_decimals(self):
        peaks = get_peaks(
            [0.0, 0.123, 0.246],
            [0.0, 1.0, 0.0],
            x=[0.0, 1000.0],
            nterms=3,
        )

        self.assertEqual(peaks.freq_decimals, 6)

    def test_save_writes_tsv(self):
        peaks = get_peaks(
            [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], cond=[2.0, 3.0, 4.0]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "peaks.tsv"
            peaks.save(path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0], "rank\tfreq\tpower\tcond")
        self.assertEqual(lines[1].split("\t"), ["1", "1.00", "1.000", "3.000"])

    def test_numpy_float32_and_float64_inputs_work(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                freq = np.array([0.0, 1.0, 2.0], dtype=dtype)
                power = np.array([0.0, 1.0, 0.0], dtype=dtype)
                peaks = get_peaks(freq, power)
                self.assertEqual(len(peaks), 1)
                self.assertAlmostEqual(float(peaks[0].freq), 1.0)
                self.assertAlmostEqual(float(peaks[0].power), 1.0)

    def test_invalid_inputs_raise_value_error(self):
        with self.assertRaises(ValueError):
            get_peaks([[0.0, 1.0, 2.0]], [[0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            get_peaks([0.0, 1.0], [0.0])
        with self.assertRaises(ValueError):
            get_peaks([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            get_peaks([0.0, math.nan, 2.0], [0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            get_peaks([0.0, 1.0, 2.0], [0.0, "bad", 0.0])


try:
    from mpmath import mp
except ImportError:  # pragma: no cover - exercised only without mpmath
    mp = None


@unittest.skipIf(mp is None, "mpmath is not installed")
class MpmathPeakUtilityTests(unittest.TestCase):
    def test_mpmath_inputs_keep_mpf_arithmetic(self):
        freq = [mp.mpf(1), mp.mpf(2), mp.mpf(3), mp.mpf(4)]
        power = [mp.mpf(10) - (x - mp.mpf("2.25")) ** 2 for x in freq]

        peaks = get_peaks(freq, power)

        self.assertIsInstance(peaks[0].freq, type(mp.mpf(0)))
        self.assertEqual(peaks[0].freq, mp.mpf("2.25"))
        self.assertEqual(peaks[0].power, mp.mpf(10))

    def test_mpmath_native_path_matches_reference(self):
        with mp.workprec(106):
            freq = [mp.mpf(i) / 16 for i in range(256)]
            power = [
                mp.sin(mp.mpf("2.7") * x)
                + mp.mpf("0.25") * mp.cos(mp.mpf("13.0") * x)
                for x in freq
            ]
            power[77] = mp.nan
            cond = [mp.mpf(2) + x + x * x for x in freq]
            cond[120] = mp.nan

            native = get_peaks(
                freq, power, cond=cond, threshold=mp.mpf("0.8"), num_peaks=None
            )
            reference = _reference_get_peaks(
                freq, power, cond=cond, threshold=mp.mpf("0.8")
            )

            self.assertEqual(len(native), len(reference))
            for native_peak, ref_peak in zip(native, reference):
                self.assertLess(abs(native_peak.freq - ref_peak.freq), mp.mpf("1e-25"))
                self.assertLess(abs(native_peak.power - ref_peak.power), mp.mpf("1e-25"))
                if mp.isnan(ref_peak.cond):
                    self.assertTrue(mp.isnan(native_peak.cond))
                else:
                    self.assertLess(abs(native_peak.cond - ref_peak.cond), mp.mpf("1e-25"))


if __name__ == "__main__":
    unittest.main()
