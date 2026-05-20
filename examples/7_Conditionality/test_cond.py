#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from toeplitz_ls import tls, tlsf


FMAX = 4.0
NTERMS = 8
OVERSAMPLING = 5
SOLVER = "levinson"
NORMALIZATION = "standard"
MIN_MEASUREMENTS = 50
LOG10_THRESHOLDS = np.round(np.arange(20.0, -0.000001, -0.125), 2)
MODULES = {
    "tls": tls,
    "tlsf": tlsf,
}


def load_dat_file(path: str | Path, precision: str):
    arr = np.loadtxt(path, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns, got {arr.shape[1]}")

    if arr.shape[0] < MIN_MEASUREMENTS:
        return None

    y_dtype = np.float64 if precision == "tls" else np.float32

    x = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(arr[:, 1], dtype=y_dtype)

    return x, y


def best_frequencies_for_thresholds(frequency, power, cond):
    power = np.asarray(power)
    cond = np.asarray(cond)

    valid_base = np.isfinite(power) & np.isfinite(cond) & (cond >= 1.0)
    log10_cond = np.full(cond.shape, np.nan, dtype=np.float64)
    log10_cond[valid_base] = np.log10(cond[valid_base])

    best_frequencies = []
    for log10_threshold in LOG10_THRESHOLDS:
        valid = valid_base & (log10_cond <= log10_threshold)

        if not np.any(valid):
            best_frequencies.append(0.0)
            continue

        valid_indices = np.flatnonzero(valid)
        best_idx = valid_indices[np.argmax(power[valid])]
        best_frequencies.append(float(frequency[best_idx]))

    return best_frequencies


def process_one_file(path: Path, precision: str):
    try:
        loaded = load_dat_file(path, precision)
        if loaded is None:
            return None

        x, y = loaded
        module = MODULES[precision]

        frequency, power, cond = module.autopower(
            x,
            y,
            fmax=FMAX,
            nterms=NTERMS,
            oversampling=OVERSAMPLING,
            solver=SOLVER,
            normalization=NORMALIZATION,
            backend="pswf43",
            autonan=False,
        )

        best_frequencies = best_frequencies_for_thresholds(frequency, power, cond)

        return path.name, path.stem, best_frequencies

    except Exception as exc:
        print(f"Error processing {path.name} with {precision}: {exc}")
        return None


def progress(iterator, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc=desc, unit="file")
    except ImportError:
        return iterator


def write_threshold_outputs(results, output_dir: Path, precision: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    open_files = []
    try:
        for log10_threshold in LOG10_THRESHOLDS:
            output_path = (
                output_dir / f"{precision}_d{NTERMS}_{log10_threshold:.2f}.tsv"
            )
            open_files.append(output_path.open("w", encoding="utf-8"))

        for _filename, stem, best_frequencies in results:
            for output_file, best_freq in zip(open_files, best_frequencies):
                output_file.write(f"{stem}\t{best_freq:.6f}\n")
    finally:
        for output_file in open_files:
            output_file.close()


def compute_condition_threshold_outputs(
    phot_dir: str | Path = "./phot",
    output_dir: str | Path = "./out_cond",
    precisions: tuple[str, ...] = ("tls", "tlsf"),
    max_workers: int = 24,
):
    phot_dir = Path(phot_dir)
    output_dir = Path(output_dir)

    files = sorted(phot_dir.glob("*.dat"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .dat files found in {phot_dir}")

    for precision in precisions:
        results = []
        skipped = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            iterator = executor.map(
                lambda path: process_one_file(path, precision),
                files,
            )

            for result in progress(
                iterator,
                total=len(files),
                desc=f"Processing {precision}",
            ):
                if result is None:
                    skipped += 1
                else:
                    results.append(result)

        write_threshold_outputs(results, output_dir, precision)
        print(
            f"{precision}: completed {len(results)} files, skipped {skipped} files, "
            f"wrote {len(LOG10_THRESHOLDS)} threshold outputs"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute TLS/TLSF best frequencies for condition-threshold sweeps."
    )
    parser.add_argument("--phot-dir", default="./phot")
    parser.add_argument("--output-dir", default="./out_cond")
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument(
        "--precision",
        choices=("tls", "tlsf", "both"),
        default="both",
        help="Run double precision, single precision, or both.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precisions = ("tls", "tlsf") if args.precision == "both" else (args.precision,)
    compute_condition_threshold_outputs(
        phot_dir=args.phot_dir,
        output_dir=args.output_dir,
        precisions=precisions,
        max_workers=args.max_workers,
    )
