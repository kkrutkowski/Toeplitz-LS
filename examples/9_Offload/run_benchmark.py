#!/usr/bin/env python3
"""Throughput and accuracy benchmark runner for Toeplitz-LS GPU offload backends."""

from __future__ import annotations

import os

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
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from methods import OffloadMethod, build_registry, make_frequency_grid  # noqa: E402
from toeplitz_ls.offload import sycl as _sycl  # noqa: E402

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
    best_power: float
    failed_spectrum: bool
    user_cpu_seconds: float
    nf: int


@dataclass(frozen=True)
class PassResult:
    rows: list[FileResult]
    status: str
    wall_seconds: float
    user_cpu_seconds: float
    total_nf: int


def load_dat_file(path: Path) -> LightCurve | None:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3 or data.shape[0] < MIN_MEASUREMENTS:
        return None
    return LightCurve(
        filename=path.name,
        source_id=path.stem,
        t=np.ascontiguousarray(data[:, 0], dtype=np.float64),
        y=np.ascontiguousarray(data[:, 1], dtype=np.float64),
        dy=np.ascontiguousarray(data[:, 2], dtype=np.float64),
    )


def load_curves(phot_dir: Path, dataset_limit: int, *, load_workers: int = 8):
    paths = sorted(phot_dir.glob("*.dat"), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No .dat files found in {phot_dir}")
    selected = paths if dataset_limit == 0 else paths[:dataset_limit]
    curves: list[LightCurve] = []
    skipped = 0
    if load_workers > 1 and len(selected) > 256:
        with ThreadPoolExecutor(max_workers=load_workers) as ex:
            for curve in ex.map(load_dat_file, selected):
                if curve is None:
                    skipped += 1
                else:
                    curves.append(curve)
    else:
        for path in selected:
            curve = load_dat_file(path)
            if curve is None:
                skipped += 1
            else:
                curves.append(curve)
    return selected, curves, skipped


def _thread_user_time() -> float:
    if hasattr(resource, "RUSAGE_THREAD"):
        return resource.getrusage(resource.RUSAGE_THREAD).ru_utime
    return time.thread_time()


def _compute_one(
    curve: LightCurve,
    method: OffloadMethod,
    nterms: int,
    *,
    fmax: float,
    oversampling: float,
    use_dy: bool,
) -> FileResult:
    grid = make_frequency_grid(curve.t, fmax=fmax, oversampling=oversampling, nterms=nterms)
    dy = curve.dy if use_dy else None
    started = _thread_user_time()
    power = method.power(
        grid.Nf,
        grid.df,
        grid.f0,
        curve.t,
        curve.y,
        dy,
        nterms=nterms,
        normalization=method.normalization,
        frequency=grid.frequency,
    )
    user_cpu = _thread_user_time() - started

    power = np.asarray(power)
    finite = np.isfinite(power)
    if not np.any(finite):
        return FileResult(curve.filename, curve.source_id, 0.0, 0.0, True, user_cpu, grid.Nf)
    indices = np.flatnonzero(finite)
    best_idx = indices[int(np.argmax(power[finite]))]
    return FileResult(
        filename=curve.filename,
        source_id=curve.source_id,
        best_frequency=float(grid.frequency[best_idx]),
        best_power=float(power[best_idx]),
        failed_spectrum=False,
        user_cpu_seconds=user_cpu,
        nf=grid.Nf,
    )


def run_pass(
    method: OffloadMethod,
    nterms: int,
    curves: list[LightCurve],
    *,
    max_workers: int,
    fmax: float,
    oversampling: float,
    use_dy: bool,
    warmup_count: int,
    progress_label: str,
    show_progress: bool,
) -> PassResult:
    if warmup_count > 0 and curves:
        n_warm = min(max(warmup_count, max_workers * 2), len(curves))
        with ThreadPoolExecutor(max_workers=max_workers) as warm_ex:
            list(
                warm_ex.map(
                    lambda wc: method.warmup(
                        wc.t,
                        wc.y,
                        wc.dy if use_dy else None,
                        nterms,
                        fmax=fmax,
                        oversampling=oversampling,
                    ),
                    curves[:n_warm],
                )
            )

    rows: list[FileResult] = []
    pass_cpu = 0.0
    total_nf = 0
    iterator = iter(curves)
    pending = set()
    progress = tqdm(
        total=len(curves),
        desc=progress_label,
        unit="lc",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def submit_more(executor):
        while len(pending) < max_workers * 2:
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

    wall_start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            submit_more(executor)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    res = fut.result()
                    rows.append(res)
                    pass_cpu += res.user_cpu_seconds
                    total_nf += res.nf
                progress.update(len(finished))
                submit_more(executor)
    finally:
        wall_seconds = time.perf_counter() - wall_start
        progress.close()

    rows.sort(key=lambda r: r.filename)
    return PassResult(
        rows=rows,
        status="complete",
        wall_seconds=wall_seconds,
        user_cpu_seconds=pass_cpu,
        total_nf=total_nf,
    )


def write_output(
    output_path: Path,
    *,
    method_name: str,
    nterms: int,
    device_name: str,
    selected_files: int,
    eligible_files: int,
    result: PassResult,
    use_dy: bool,
    max_workers: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed = sum(r.failed_spectrum for r in result.rows)
    lc_per_sec = len(result.rows) / max(result.wall_seconds, 1e-9)
    mfreq_per_sec = (result.total_nf / 1e6) / max(result.wall_seconds, 1e-9)
    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# method\t{method_name}\n")
        out.write(f"# nterms\t{nterms}\n")
        out.write(f"# status\t{result.status}\n")
        out.write(f"# device\t{device_name}\n")
        out.write(f"# max_workers\t{max_workers}\n")
        out.write(f"# selected_files\t{selected_files}\n")
        out.write(f"# eligible_files\t{eligible_files}\n")
        out.write(f"# completed_files\t{len(result.rows)}\n")
        out.write(f"# failed_spectra\t{failed}\n")
        out.write(f"# use_dy\t{str(use_dy).lower()}\n")
        out.write(f"# wall_seconds\t{result.wall_seconds:.6f}\n")
        out.write(f"# curves_per_second\t{lc_per_sec:.3f}\n")
        out.write(f"# mfreqs_per_second\t{mfreq_per_sec:.3f}\n")
        out.write(f"# power_user_cpu_hours\t{result.user_cpu_seconds / 3600.0:.12g}\n")
        out.write(f"# cumulative_power_user_cpu_hours\t{result.user_cpu_seconds / 3600.0:.12g}\n")
        out.write("source_id\tbest_frequency\tbest_power\n")
        for row in result.rows:
            out.write(f"{row.source_id}\t{row.best_frequency:.12g}\t{row.best_power:.12g}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark Toeplitz-LS GPU offload vs CPU on light curve datasets.")
    parser.add_argument("--methods", default="sycl_tlsh,sycl_tlsf,sycl_tlsdf,sycl_tls")
    parser.add_argument("--phot-dir", default="/home/krutkowski/Pulpit/LMC_ECL/phot")
    parser.add_argument("--output-dir", default="./out_offload")
    parser.add_argument("--device", type=int, default=0, help="SYCL device ID from list_devices().")
    parser.add_argument("--backend", default="pswf43", choices=("pswf43", "pswf21"))
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--nterms", default="1,3", help="Comma-separated harmonic orders, e.g. 1,3.")
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--oversampling", type=float, default=5.0)
    parser.add_argument("--use-dy", action="store_true")
    parser.add_argument("--warmup", type=int, default=4, help="Unmeasured warmup curves before timing each pass.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    devices = _sycl.list_devices(print_table=True)
    if args.list_devices:
        return
    dev_desc = next((d for d in devices if d.id == args.device), None)
    dev_label = f"[{dev_desc.id}] {dev_desc.name}" if dev_desc is not None else f"[{args.device}] unknown"

    registry = build_registry(device=args.device, pswf_backend=args.backend)
    if args.methods.strip().lower() == "all":
        method_names = list(registry)
    else:
        method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in method_names if m not in registry]
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}; choices: {list(registry)}")
    nterms_list = [int(x.strip()) for x in str(args.nterms).split(",") if x.strip()]

    t_load0 = time.perf_counter()
    selected, curves, skipped = load_curves(Path(args.phot_dir), args.dataset_limit)
    t_load = time.perf_counter() - t_load0
    print(
        f"Loaded {len(curves)} eligible curves ({skipped} < {MIN_MEASUREMENTS} pts) "
        f"from {len(selected)} files in {t_load:.2f}s."
    )

    out_dir = Path(args.output_dir)
    for nterms in nterms_list:
        for mname in method_names:
            method = registry[mname]
            res = run_pass(
                method,
                nterms,
                curves,
                max_workers=args.max_workers,
                fmax=args.fmax,
                oversampling=args.oversampling,
                use_dy=args.use_dy,
                warmup_count=args.warmup,
                progress_label=f"{mname} d={nterms}",
                show_progress=not args.no_progress,
            )
            out_path = out_dir / f"{mname}_nterms{nterms}.tsv"
            write_output(
                out_path,
                method_name=mname,
                nterms=nterms,
                device_name=dev_label if method.backend_family == "sycl" else "CPU",
                selected_files=len(selected),
                eligible_files=len(curves),
                result=res,
                use_dy=args.use_dy,
                max_workers=args.max_workers,
            )
            lc_s = len(res.rows) / max(res.wall_seconds, 1e-9)
            mf_s = (res.total_nf / 1e6) / max(res.wall_seconds, 1e-9)
            print(
                f"{mname:22s} d={nterms:2d}: wall={res.wall_seconds:7.2f}s | "
                f"{lc_s:8.1f} lc/s | {mf_s:7.2f} Mfreq/s | "
                f"cpu={res.user_cpu_seconds:7.2f}s | failed={sum(r.failed_spectrum for r in res.rows)}"
            )


if __name__ == "__main__":
    main()
