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
        scalar = get_peaks(list(freq), list(power), cond=list(cond))

        self.assertEqual(len(fast), len(scalar))
        for fast_peak, scalar_peak in zip(fast, scalar):
            self.assertAlmostEqual(float(fast_peak.freq), float(scalar_peak.freq))
            self.assertAlmostEqual(float(fast_peak.power), float(scalar_peak.power))
            if math.isnan(float(scalar_peak.cond)):
                self.assertTrue(math.isnan(float(fast_peak.cond)))
            else:
                self.assertAlmostEqual(float(fast_peak.cond), float(scalar_peak.cond))

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


if __name__ == "__main__":
    unittest.main()
