import argparse
import sys
import time
from pathlib import Path

import nifty_ls
import numpy as np
from mpmath import mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toeplitz-ls"))

from toeplitz_ls import tls, tlsdd, tlsf


mp.prec = 106
MPF_TYPE = type(mp.mpf(0))
MEDIAN_DIFF_MAX_POINTS = 1 << 16


def make_signal(M=1000):
    rng = np.random.default_rng(seed=314159)
    t = np.sort(rng.uniform(0.0, 120.0, M)).astype(np.float64)
    dy = rng.uniform(0.08, 0.30, M).astype(np.float64)

    y_clean = (
        1.35 * np.sin(2.0 * np.pi * 0.173828125 * t + 0.20)
        + 0.85 * np.cos(2.0 * np.pi * 0.7421875 * t - 0.55)
        + 0.40 * np.sin(2.0 * np.pi * 1.53125 * t + 1.15)
    )
    y = (y_clean + rng.normal(0.0, dy)).astype(np.float64)
    return t, y, dy


def timed(label, fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return label, elapsed, value


def diff_stats(candidate, reference):
    n = len(candidate)
    median_count = min(n, MEDIAN_DIFF_MAX_POINTS)
    median_start = (n - median_count) // 2
    median_stop = median_start + median_count
    median_diff = [
        abs(candidate[i] - reference[i]) for i in range(median_start, median_stop)
    ]
    median_diff.sort()

    if median_count % 2:
        median = median_diff[median_count // 2]
    else:
        median = (
            median_diff[median_count // 2 - 1] + median_diff[median_count // 2]
        ) / 2

    total = mp.mpf(0)
    maximum = mp.mpf(0)
    for c, r in zip(candidate, reference):
        value = abs(c - r)
        total += value
        if value > maximum:
            maximum = value

    return median, total / n, maximum


def as_mpf_power(power):
    return [
        value if isinstance(value, MPF_TYPE) else mp.mpf(float(value))
        for value in power
    ]


def call_tlsf(t, y, dy, f0, df, nf, nterms, backend, solver):
    return tlsf.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    ).astype(np.float64)


def call_tls(t, y, dy, f0, df, nf, nterms, backend, solver):
    return tls.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    )


def call_tlsdd(t, y, dy, f0, df, nf, nterms, backend, solver):
    return tlsdd.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    )


def run_nifty(t, y, dy, f0, df, nf, nterms):
    fmax = f0 + df * (nf - 1)
    result = nifty_ls.lombscargle(
        t,
        y,
        dy=dy,
        fmin=f0,
        fmax=fmax,
        Nf=nf,
        center_data=True,
        fit_mean=True,
        normalization="standard",
        assume_sorted_t=True,
        nterms=nterms,
        nthreads=1,
    )
    if result.Nf != nf or result.fmin != f0 or result.df != df:
        raise RuntimeError(
            "Nifty frequency grid mismatch: "
            f"got fmin={result.fmin}, df={result.df}, Nf={result.Nf}"
        )
    return np.asarray(result.power, dtype=np.float64)


def print_header(nterms, f0, df):
    print()
    print(f"nterms={nterms}, f0={f0:.17g}, df={df:.17g}")
    print(
        f"{'N':>8s}  {'method':14s}  {'time [s]':>10s}  "
        f"{'median abs':>12s}  {'average abs':>12s}  {'maximum abs':>12s}"
    )
    print("-" * 80)


def print_row(nf, method, elapsed, candidate, reference):
    med, avg, max_diff = diff_stats(candidate, reference)
    print(
        f"{nf:8d}  {method:14s}  {elapsed:10.4f}  "
        f"{mp.nstr(med, 2):>12s}  "
        f"{mp.nstr(avg, 2):>12s}  "
        f"{mp.nstr(max_diff, 2):>12s}"
    )


def benchmark(args):
    t, y, dy = make_signal(M=args.M)

    # Both f0 and df are binary fractions; df is exactly a power of two.
    f0 = 2.0**-7
    df = 2.0**-15

    c_methods = [
        (
            "tlsdd-PSWF-L",
            lambda nf, nt: call_tlsdd(t, y, dy, f0, df, nf, nt, 0, 1),
        ),
        ("tls-LRA-L", lambda nf, nt: call_tls(t, y, dy, f0, df, nf, nt, 1, 1)),
        ("tls-LRA-Z", lambda nf, nt: call_tls(t, y, dy, f0, df, nf, nt, 1, 2)),
        ("tls-PSWF-L", lambda nf, nt: call_tls(t, y, dy, f0, df, nf, nt, 0, 1)),
        ("tls-PSWF-Z", lambda nf, nt: call_tls(t, y, dy, f0, df, nf, nt, 0, 2)),
        (
            "tlsf-LRA-L",
            lambda nf, nt: call_tlsf(t, y, dy, f0, df, nf, nt, 1, 1),
        ),
        (
            "tlsf-LRA-Z",
            lambda nf, nt: call_tlsf(t, y, dy, f0, df, nf, nt, 1, 2),
        ),
        (
            "tlsf-PSWF-L",
            lambda nf, nt: call_tlsf(t, y, dy, f0, df, nf, nt, 0, 1),
        ),
        (
            "tlsf-PSWF-Z",
            lambda nf, nt: call_tlsf(t, y, dy, f0, df, nf, nt, 0, 2),
        ),
    ]

    for nterms in (1, 3):  # , 8
        print_header(nterms, f0, df)

        for k in range(args.min_power, args.max_power + 1):
            nf = 1 << k

            print()

            ref_label, ref_time, reference = timed(
                "tlsdd-LRA-L",
                lambda nf=nf, nterms=nterms: call_tlsdd(
                    t, y, dy, f0, df, nf, nterms, 1, 1
                ),
            )
            print_row(nf, ref_label, ref_time, reference, reference)

            for method, runner in c_methods:
                label, elapsed, candidate = timed(
                    method,
                    lambda runner=runner, nf=nf, nterms=nterms: runner(nf, nterms),
                )
                candidate = as_mpf_power(candidate)
                print_row(nf, label, elapsed, candidate, reference)

            label, elapsed, candidate = timed(
                "nifty",
                lambda nf=nf, nterms=nterms: run_nifty(t, y, dy, f0, df, nf, nterms),
            )
            candidate = as_mpf_power(candidate)
            print_row(nf, label, elapsed, candidate, reference)

            if args.blank_lines:
                print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark chi2per shared-library methods against nifty_ls. "
            "Accuracy is measured against tlsdd-LRA."
        )
    )
    parser.add_argument("--M", type=int, default=1000)
    parser.add_argument("--min-power", type=int, default=10)
    parser.add_argument("--max-power", type=int, default=20)
    parser.add_argument(
        "--blank-lines",
        action="store_true",
        help="Print a blank line after each N block.",
    )
    args = parser.parse_args()

    if args.M <= 0:
        raise SystemExit("--M must be positive")
    if args.min_power < 5 or args.max_power < args.min_power:
        raise SystemExit("Require 5 <= --min-power <= --max-power")

    benchmark(args)


if __name__ == "__main__":
    main()
