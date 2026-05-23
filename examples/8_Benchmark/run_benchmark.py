#!/usr/bin/env python3
"""Run dataset periodogram comparisons on a common per-source grid."""

from __future__ import annotations

import os

# Avoid nested BLAS/FFT parallelism when the dataset itself is threaded.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import resource
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from methods import METHODS, Method, make_frequency_grid  # noqa: E402

MIN_MEASUREMENTS = 50


@dataclass(frozen=True)
class LightCurve:
    filename: str
    source_id: str
    t: np.ndarray
    y: np.ndarray
    dy: np.ndarray


@dataclass(frozen=True)
class FileResult:
    filename: str
    source_id: str
    best_frequency: float
    failed_spectrum: bool
    user_cpu_seconds: float


@dataclass(frozen=True)
class PassResult:
    rows: list[FileResult]
    status: str
    user_cpu_seconds: float


def load_dat_file(path: str | Path) -> LightCurve | None:
    path = Path(path)
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns, got {data.shape[1]}")
    if data.shape[0] < MIN_MEASUREMENTS:
        return None
    return LightCurve(
        filename=path.name,
        source_id=path.stem,
        t=np.ascontiguousarray(data[:, 0], dtype=np.float64),
        y=np.ascontiguousarray(data[:, 1], dtype=np.float64),
        dy=np.ascontiguousarray(data[:, 2], dtype=np.float64),
    )


def load_curves(phot_dir: Path, dataset_limit: int):
    paths = sorted(phot_dir.glob("*.dat"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No .dat files found in {phot_dir}")
    selected_paths = paths if dataset_limit == 0 else paths[:dataset_limit]
    curves = []
    skipped_short = 0
    for path in selected_paths:
        curve = load_dat_file(path)
        if curve is None:
            skipped_short += 1
        else:
            curves.append(curve)
    return selected_paths, curves, skipped_short


def _thread_user_time() -> float:
    if hasattr(resource, "RUSAGE_THREAD"):
        return resource.getrusage(resource.RUSAGE_THREAD).ru_utime
    return time.thread_time()


def _compute_one(
    curve: LightCurve,
    method: Method,
    nterms: int,
    *,
    fmax: float,
    oversampling: float,
    use_dy: bool,
) -> FileResult:
    grid = make_frequency_grid(
        curve.t, fmax=fmax, oversampling=oversampling, nterms=nterms
    )
    dy = curve.dy if use_dy else None
    failed_spectrum = False

    started = _thread_user_time()
    try:
        power = method.power(
            grid.Nf,
            grid.df,
            grid.f0,
            curve.t,
            curve.y,
            dy,
            nterms=nterms,
            normalization="standard",
            frequency=grid.frequency,
        )
    except np.linalg.LinAlgError:
        if method.name != "nifty":
            raise
        power = None
        failed_spectrum = True
    finally:
        user_cpu_seconds = _thread_user_time() - started

    if failed_spectrum:
        best_frequency = 0.0
    else:
        power = np.asarray(power)
        finite = np.isfinite(power)
        if not np.any(finite):
            best_frequency = 0.0
            failed_spectrum = True
        else:
            indices = np.flatnonzero(finite)
            best_index = indices[np.argmax(power[finite])]
            best_frequency = float(grid.frequency[best_index])

    return FileResult(
        filename=curve.filename,
        source_id=curve.source_id,
        best_frequency=best_frequency,
        failed_spectrum=failed_spectrum,
        user_cpu_seconds=user_cpu_seconds,
    )


def run_pass(
    method: Method,
    nterms: int,
    curves: list[LightCurve],
    *,
    max_workers: int,
    fmax: float,
    oversampling: float,
    use_dy: bool,
    prior_user_cpu_seconds: float,
    compute_limit_seconds: float,
    progress_label: str,
    show_progress: bool,
) -> PassResult:
    rows = []
    pass_cpu_seconds = 0.0
    budget_reached = False
    iterator = iter(curves)
    pending = set()
    progress = tqdm(
        total=len(curves),
        desc=progress_label,
        unit="file",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def submit_available(executor):
        nonlocal budget_reached
        while not budget_reached and len(pending) < max_workers:
            try:
                curve = next(iterator)
            except StopIteration:
                return
            pending.add(
                executor.submit(
                    _compute_one,
                    curve,
                    method,
                    nterms,
                    fmax=fmax,
                    oversampling=oversampling,
                    use_dy=use_dy,
                )
            )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            submit_available(executor)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    result = future.result()
                    rows.append(result)
                    pass_cpu_seconds += result.user_cpu_seconds
                progress.update(len(finished))
                progress.set_postfix(
                    cpu_h=f"{pass_cpu_seconds / 3600.0:.4g}", refresh=False
                )
                if (
                    compute_limit_seconds > 0.0
                    and prior_user_cpu_seconds + pass_cpu_seconds
                    >= compute_limit_seconds
                ):
                    budget_reached = True
                submit_available(executor)
    finally:
        progress.close()

    rows.sort(key=lambda row: row.filename)
    status = "compute_limit_reached" if budget_reached else "complete"
    return PassResult(rows=rows, status=status, user_cpu_seconds=pass_cpu_seconds)


def write_output(
    output_path: Path,
    *,
    method_name: str,
    nterms: int,
    status: str,
    selected_files: int,
    eligible_files: int,
    rows: list[FileResult],
    use_dy: bool,
    pass_user_cpu_seconds: float,
    cumulative_user_cpu_seconds: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_spectra = sum(row.failed_spectrum for row in rows)
    with output_path.open("w", encoding="utf-8") as output:
        output.write(f"# method\t{method_name}\n")
        output.write(f"# nterms\t{nterms}\n")
        output.write(f"# status\t{status}\n")
        output.write(f"# selected_files\t{selected_files}\n")
        output.write(f"# eligible_files\t{eligible_files}\n")
        output.write(f"# completed_files\t{len(rows)}\n")
        output.write(f"# failed_spectra\t{failed_spectra}\n")
        output.write(f"# use_dy\t{str(use_dy).lower()}\n")
        output.write(f"# power_user_cpu_hours\t{pass_user_cpu_seconds / 3600.0:.12g}\n")
        output.write(
            "# cumulative_power_user_cpu_hours\t"
            f"{cumulative_user_cpu_seconds / 3600.0:.12g}\n"
        )
        output.write("source_id\tbest_frequency\n")
        for row in rows:
            output.write(f"{row.source_id}\t{row.best_frequency:.12g}\n")


def parse_methods(value: str, registry) -> list[str]:
    if value.strip().lower() == "all":
        return list(registry)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in registry]
    if unknown:
        choices = ", ".join(registry)
        raise ValueError(f"unknown method(s): {', '.join(unknown)}; choices are {choices}")
    if not names:
        raise ValueError("at least one method must be selected")
    return names


def run_benchmark(args, registry=METHODS):
    phot_dir = Path(args.phot_dir)
    output_dir = Path(args.output_dir)
    selected_paths, curves, skipped_short = load_curves(phot_dir, args.dataset_limit)
    method_names = parse_methods(args.methods, registry)
    compute_limit_seconds = args.compute_limit * 3600.0
    nterms_values = (
        [args.nterms] if args.nterms is not None else range(1, args.nterms_max + 1)
    )

    print(
        f"Loaded {len(curves)} eligible curves from {len(selected_paths)} selected files "
        f"({skipped_short} shorter than {MIN_MEASUREMENTS} measurements)."
    )
    written = []
    cumulative_seconds = dict.fromkeys(method_names, 0.0)
    stopped_methods = set()

    for nterms in nterms_values:
        for method_name in method_names:
            if method_name in stopped_methods:
                continue
            method = registry[method_name]
            output_path = output_dir / f"{method_name}_nterms{nterms}.tsv"
            if not method.supports(nterms):
                write_output(
                    output_path,
                    method_name=method_name,
                    nterms=nterms,
                    status="unsupported",
                    selected_files=len(selected_paths),
                    eligible_files=len(curves),
                    rows=[],
                    use_dy=args.use_dy,
                    pass_user_cpu_seconds=0.0,
                    cumulative_user_cpu_seconds=cumulative_seconds[method_name],
                )
                written.append(output_path)
                print(f"{method_name} nterms={nterms}: unsupported")
                continue

            # Always finish the baseline order so every selected method has data.
            effective_compute_limit_seconds = (
                0.0 if nterms == 1 else compute_limit_seconds
            )
            result = run_pass(
                method,
                nterms,
                curves,
                max_workers=args.max_workers,
                fmax=args.fmax,
                oversampling=args.oversampling,
                use_dy=args.use_dy,
                prior_user_cpu_seconds=cumulative_seconds[method_name],
                compute_limit_seconds=effective_compute_limit_seconds,
                progress_label=f"{method_name} nterms={nterms}",
                show_progress=not args.no_progress,
            )
            cumulative_seconds[method_name] += result.user_cpu_seconds
            write_output(
                output_path,
                method_name=method_name,
                nterms=nterms,
                status=result.status,
                selected_files=len(selected_paths),
                eligible_files=len(curves),
                rows=result.rows,
                use_dy=args.use_dy,
                pass_user_cpu_seconds=result.user_cpu_seconds,
                cumulative_user_cpu_seconds=cumulative_seconds[method_name],
            )
            written.append(output_path)
            print(
                f"{method_name} nterms={nterms}: {result.status}, "
                f"{len(result.rows)}/{len(curves)} curves, "
                f"{result.user_cpu_seconds / 3600.0:.6g} CPU-hours "
                f"({cumulative_seconds[method_name] / 3600.0:.6g} cumulative)"
            )
            if (
                compute_limit_seconds > 0.0
                and cumulative_seconds[method_name] >= compute_limit_seconds
            ):
                stopped_methods.add(method_name)
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark FastChi2-compatible periodogram implementations on light curves."
    )
    parser.add_argument("--methods", default="all")
    parser.add_argument("--phot-dir", default="./phot")
    parser.add_argument("--index-path", default="./index.tsv")
    parser.add_argument("--output-dir", default="./out")
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument(
        "--nterms",
        type=int,
        default=None,
        help="Run only this harmonic order instead of sweeping 1..--nterms-max.",
    )
    parser.add_argument("--nterms-max", type=int, default=12)
    parser.add_argument("--compute-limit", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--oversampling", type=float, default=5.0)
    parser.add_argument("--use-dy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset_limit < 0:
        parser.error("--dataset-limit must be non-negative")
    if args.nterms_max < 1:
        parser.error("--nterms-max must be positive")
    if args.nterms is not None and args.nterms < 1:
        parser.error("--nterms must be positive")
    if args.compute_limit < 0.0:
        parser.error("--compute-limit must be non-negative")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.fmax <= 0.0 or args.oversampling <= 0.0:
        parser.error("--fmax and --oversampling must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
