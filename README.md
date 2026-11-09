# Toeplitz-LS

Reference implementation of the multiharmonic least-squares periodogram
(AoVMH(W) / FastChi²) using a blocked NuFFT and a fast Toeplitz solver for
complex coefficients.

## Installation

The current release is built from source. The package requires Python >= 3.8
and NumPy >= 1.17.3; the optional `tlsdd` extra requires mpmath >= 1.0.

```console
git clone https://github.com/kkrutkowski/Toeplitz-LS.git --depth=1
cd ./Toeplitz-LS
make native        # or 'make generic' for a portable baseline
cd ./toeplitz-ls
pip3 install .     # use 'pip3 install ".[tlsdd]"' for the mpmath wrappers
```

## Supported platforms

Linux, Windows, and macOS are supported on x86-64 and ARM64.

## Runtime dispatch

Release builds bundle more than one native library. At import,
`toeplitz_ls` probes the CPU and loads the widest variant it can execute,
falling back to the baseline library; the choice is made once, so a
periodogram call pays nothing for it. `TOEPLITZ_LS_DISPATCH` forces a specific
variant, and `TOEPLITZ_LS_DISPATCH=list` prints the available variants.

## Usage

The package exposes two periodogram families. The FastChi2/AoVMH modules
(`tlsf`, `tls`, `tlsdd`) use the Toeplitz least-squares solve. The IHS modules
(`ihsf`, `ihs`, `ihsdd`) use the incoherent harmonic sum: one NuFFT per
harmonic, no linear solve, and only the `raw`, `nll`, or `default`
statistic. Each
family has single-, double-, and double-double-precision variants. The single-
and double-precision variants accept and return NumPy arrays. The double-double
variants rely on compensated arithmetic and operate on a 106-bit mantissa; they
use mpmath's `mpf` type to bridge the gap between the library's internal
representation and Python's native numeric types. The `get_peaks` helper turns
a periodogram into a sorted, interpolated peak list for printing or saving.

## Normalizations

For the FastChi2 modules (`tlsf`, `tls`, `tlsdd`), `power()` and `autopower()`
accept `normalization=`:

| value | output | notes |
|---|---|---|
| `"standard"` (default) | coefficient of determination `R^2` | |
| `"anova"` | ANOVA / `F` statistic | Schwarzenberg-Czerny's ANOVA convention |
| `"nll"` | natural negative log false-alarm probability | `-ln(FAP)` |

For `nterms = d` harmonics:

- `"standard"` is the coefficient of determination `R^2` of the full model.
  Values at or above 1 are numerical artefacts of an ill-conditioned solve and
  are reported as 0.
- `"anova"` is the ANOVA `F` statistic in Schwarzenberg-Czerny's convention.
- `"nll"` is the natural-log negative false-alarm probability, `-ln(FAP)`,
  computed from the direct Pochhammer/incomplete-beta statistic with its
  finite-sample correction. `Neff = (sum w)^2 / sum(w^2)` with `w = 1/dy^2`.

The `cond` array returned when `autonan=False` is the Toeplitz condition upper
bound `(2*nterms + 1) * P`, where `P` is the Cybenko reflection-coefficient
product.

## IHS periodogram

The `ihsf`, `ihs`, and `ihsdd` modules expose the same `power()` and
`autopower()` call shape, but no `solver`, no `autonan`, and no condition
array:

```python
from toeplitz_ls import ihs

frequency, power = ihs.autopower(t, y, dy=None, nterms=3, normalization="nll")
```

IHS is a simplified formulation of a similar periodogram. It assumes that the
harmonic frequencies are independent (a regression from the standard
least-squares periodogram) but not orthogonal. It evaluates `d` NuFFTs instead
of `3d` and replaces the `O(d^2)` normal-equation solve with an `O(d)` power
accumulation, so it is at least 3x less computationally expensive. In practice
it is usually slightly less accurate than FastChi2.

The three statistics are:

- `"raw"`: a direct display statistic built from the normalized per-harmonic
  Rayleigh powers. It is not a calibrated false-alarm probability, and for
  `nterms > 1` it is not a bounded `R^2`; it may exceed 1.
- `"nll"`: the negative log false-alarm probability. Each harmonic's Rayleigh
  power is passed through Wilkie's (1983) finite-sample correction, and the
  harmonic sum is then passed through the Erlang correction.
  `Neff = (sum |wy|)^2 / sum(wy^2)`, where `wy = w * (y - ymean)`. Use this
  statistic when a reliable significance estimate is needed.
- `"default"` (the default): the same initial formula and Erlang correction as
  `"nll"`, but without the finite-sample Wilkie correction. This still produces
  NLL estimates, just less accurately.

IHS is not suitable when `nterms^4 > Neff`, where `Neff` is the raw-weight
effective sample size from the FastChi2/TLS path; the Python wrappers emit a
`RuntimeWarning` in that case. Use the FastChi2 modules or reduce `nterms`.

## Free-threaded (GIL-free) Python support

The library is written almost exclusively in C and called through `ctypes`, so
independent `power()`/`autopower()` calls release the GIL and can run in
parallel from a `ThreadPoolExecutor`. Multithreading is therefore expected to
give major performance gains even on regular CPython; for optimum performance,
use a free-threaded build (PEP 703, optional since Python 3.13).

mpmath is not thread-safe by default. When using `tlsdd` in a multithreaded
context, each worker should keep its own `mp` context via `mpmath.workprec()`
rather than mutating the global `mp.prec`.

## NaN handling

When the system matrix at a given frequency is numerically unstable, the solver
returns NaN rather than raising an exception. Use `np.nanargmax()` and
`np.nanmax()` in place of their standard counterparts to safely recover the
peak frequency and power without short-circuiting on degenerate bins.

For manual filtering, call `power()` or `autopower()` with `autonan=False` to
receive rough conditionality estimates alongside the periodogram. These
estimates are solver-specific: LDLT reports a diagonal-ratio estimate, while
the three Toeplitz solvers (`levinson`, `zohar`, and `bareiss`) return a
reflection-coefficient-based upper bound `(2*nterms + 1) * P`. The Toeplitz
estimates are generally much higher (and more reliable) for the same
frequencies, so threshold values should be chosen with the solver family in
mind.

### Example usage of `tls` and `tlsf`

```python
import numpy as np
import matplotlib.pyplot as plt
from toeplitz_ls import tls

rng = np.random.default_rng(seed=123)
N = 1000
t = np.sort(rng.uniform(0, 1000, size=N))
y = np.sin(50 * t) + 0.5 * np.sin(150 * t) + 1 + rng.poisson(size=N)

# Compute the generalized Scargle periodogram.
freq, r2 = tls.autopower(
    t, y, dy=None,
    fmax=12,  # default value
    nterms=1,  # generalized Scargle periodogram
    oversampling=5,  # frequency density, relative to a dense, uniform FFT
    backend="pswf43",  # or "pswf21", "lra"
    normalization="standard",  # or "anova", "nll"
)

best_idx = np.nanargmax(r2)
best_freq = freq[best_idx]
best_power = r2[best_idx]

print(f"Peak frequency: {best_freq:.6g} (R^2 = {best_power:.6g})")

plt.figure(figsize=(12, 5))
plt.plot(freq, r2, label="degree = 1", color="red")
plt.axvline(best_freq, linestyle="--", color="black", alpha=0.7)
plt.xlabel("Frequency")
plt.ylabel("Coefficient of determination")
plt.legend()
plt.tight_layout()
plt.show()
```

### Example usage with an OGLE-BLAP light curve

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
from toeplitz_ls import tlsf, get_peaks

url = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/BLAP/phot/phot_ogle4/I/OGLE-BLAP-035.dat"

with TemporaryDirectory() as tmpdir:
    data_path = Path(tmpdir) / "OGLE-BLAP-035.dat"
    urlretrieve(url, str(data_path))
    t, y, _dy = np.loadtxt(data_path, unpack=True)

t = t - t.min()

# BLAPs require high-frequency search. Ignore the photometric error estimates here.
freq, power, cond = tlsf.autopower(
    t, y, dy=None,
    fmax=100,
    nterms=3,
    oversampling=5,
    solver="levinson",  # or 'zohar', 'bareiss', 'ldlt', 'svd'
    normalization="anova",
    autonan=False,
)

peaks = get_peaks(freq, power, cond, threshold=10)
peaks.print(10)
```

## Development

The C and Python halves share one style: `.clang-format` and `ruff.toml` use the
same 160-column budget and 4-space indentation, so `make format` leaves both in
a state `make format-check` accepts.

```console
make format        # clang-format (src, include, tests) + ruff format/check (package, dispatch)
make format-check  # the same, without writing
make generic       # portable build for this machine's architecture
make native        # build for this machine
make -C tests check
```

Build artifacts (objects, staged wheel payloads, `.toolchains/`) live under
`.build/`. Nothing is committed to it.

## Performance

The hot path is a blocked NuFFT sweep with vectorized transforms, deferred
permutations, and block-sized work arrays. On a modern
x86-64 CPU, a periodogram over roughly 10^4 frequencies takes on the order of
milliseconds; the IHS path is at least 3x cheaper. The benchmark harness lives
in `tests/`.
