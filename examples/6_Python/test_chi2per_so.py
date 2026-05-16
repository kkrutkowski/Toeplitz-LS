import sys
from pathlib import Path

import numpy as np
from mpmath import mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toeplitz-ls"))

from toeplitz_ls import tls, tlsdd, tlsf


mp.prec = 106


def make_signal():
    rng = np.random.default_rng(seed=1729)
    t = np.sort(rng.uniform(0.0, 80.0, 256)).astype(np.float64)
    dy = rng.uniform(0.08, 0.35, t.size).astype(np.float64)

    y_clean = (
        1.6 * np.sin(2.0 * np.pi * 0.723 * t + 0.25)
        + 0.7 * np.cos(2.0 * np.pi * 1.187 * t - 0.4)
        + 0.25 * np.sin(2.0 * np.pi * 2.061 * t + 1.2)
    )
    y = (y_clean + rng.normal(0.0, dy)).astype(np.float64)

    f0 = 0.05
    df = 0.004
    nf = 768
    nterms = 3
    return t, y, dy, f0, df, nf, nterms


def call_tlsf(t, y, dy, f0, df, nf, nterms, backend, solver):
    power = tlsf.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    )
    return [mp.mpf(float(value)) for value in power]


def call_tls(t, y, dy, f0, df, nf, nterms, backend, solver):
    power = tls.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    )
    return [mp.mpf(float(value)) for value in power]


def call_tlsdd(t, y, dy, f0, df, nf, nterms, backend, solver):
    return tlsdd.power(
        nf, df, f0, t, y, dy, backend=backend, solver=solver, nterms=nterms
    )


def _mp_stats(values):
    values = sorted(values)
    n = len(values)
    if n % 2:
        median = values[n // 2]
    else:
        median = (values[n // 2 - 1] + values[n // 2]) / 2
    average = mp.fsum(values) / n
    maximum = values[-1]
    return median, average, maximum


def print_difference(label, candidate, reference):
    diff = [abs(c - r) for c, r in zip(candidate, reference)]
    median, average, maximum = _mp_stats(diff)
    print(
        f"{label:16s} "
        f"median={mp.nstr(median, 8):>12s}  "
        f"average={mp.nstr(average, 8):>12s}  "
        f"maximum={mp.nstr(maximum, 8):>12s}"
    )


def main():
    t, y, dy, f0, df, nf, nterms = make_signal()

    reference = call_tlsdd(t, y, dy, f0, df, nf, nterms, "lra", "levinson")

    print("FastChi2 .so vs tlsdd-LRA-Levinson")
    print(f"M={t.size}, Nf={nf}, f0={f0}, df={df}, nterms={nterms}")
    print()

    backends = {"pswf": "PSWF", "lra": "LRA"}
    callers = [
        ("tlsf", call_tlsf),
        ("tls", call_tls),
        ("tlsdd", call_tlsdd),
    ]

    for backend, backend_name in backends.items():
        for precision, caller in callers:
            for solver in ("levinson", "zohar", "bareiss", "ldlt"):
                candidate = caller(t, y, dy, f0, df, nf, nterms, backend, solver)
                print_difference(
                    f"{precision}-{backend_name}-{solver.title()}", candidate, reference
                )


if __name__ == "__main__":
    main()
