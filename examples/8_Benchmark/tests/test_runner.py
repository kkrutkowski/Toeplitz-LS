import argparse
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from compare import read_result
from methods import Method
from run_benchmark import run_benchmark


class PeakMethod(Method):
    name = "peak"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        result = np.zeros(Nf)
        result[min(2, Nf - 1)] = 1.0
        return result


class FailedNifty(Method):
    name = "nifty"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        raise np.linalg.LinAlgError("singular")


class BusyMethod(Method):
    name = "busy"

    def power(
        self, Nf, df, f0, t, y, dy=None, nterms=1, normalization="standard", *, frequency=None
    ):
        end = time.thread_time() + 0.008
        value = 0
        while time.thread_time() < end:
            value += 1
        result = np.zeros(Nf)
        result[0] = value
        return result


def make_args(phot_dir, output_dir, methods, **overrides):
    values = dict(
        phot_dir=str(phot_dir),
        output_dir=str(output_dir),
        index_path=str(phot_dir / "index.tsv"),
        methods=methods,
        dataset_limit=0,
        nterms_max=1,
        compute_limit=0.0,
        max_workers=1,
        fmax=1.0,
        oversampling=5.0,
        use_dy=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class RunnerTests(unittest.TestCase):
    def write_curves(self, directory, count=3):
        t = np.linspace(0.0, 20.0, 60)
        for index in range(count):
            y = np.sin(2 * np.pi * 0.2 * t + index)
            dy = np.full_like(t, 0.1)
            np.savetxt(directory / f"source{index}.dat", np.column_stack((t, y, dy)))

    def test_writes_metadata_and_converts_nifty_failure_to_zero(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            out = directory / "out"
            self.write_curves(directory, count=2)
            registry = {"peak": PeakMethod(), "nifty": FailedNifty()}
            run_benchmark(make_args(directory, out, "peak,nifty"), registry=registry)

            metadata, rows = read_result(out / "peak_nterms1.tsv")
            self.assertEqual(metadata["method"], "peak")
            self.assertEqual(metadata["completed_files"], "2")
            self.assertEqual(len(rows), 2)

            metadata, rows = read_result(out / "nifty_nterms1.tsv")
            self.assertEqual(metadata["failed_spectra"], "2")
            self.assertTrue(all(frequency == 0.0 for _name, frequency in rows))

    def test_cpu_limit_keeps_partial_pass(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            out = directory / "out"
            self.write_curves(directory, count=4)
            registry = {"busy": BusyMethod()}
            args = make_args(
                directory,
                out,
                "busy",
                nterms_max=3,
                compute_limit=0.000001,
            )
            written = run_benchmark(args, registry=registry)
            metadata, rows = read_result(out / "busy_nterms1.tsv")
            self.assertEqual(metadata["status"], "compute_limit_reached")
            self.assertGreaterEqual(len(rows), 1)
            self.assertLess(len(rows), 4)
            self.assertEqual([path.name for path in written], ["busy_nterms1.tsv"])


if __name__ == "__main__":
    unittest.main()
