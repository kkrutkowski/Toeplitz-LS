# Toeplitz-LS
Reference implementation of multiharmonic least-squares periodogram (AoVMH(W) / FastChi²) using blocked NuFFT and a fast Toeplitz solver for complex coefficients.

## Installation

Prebuilt wheels are the recommended installation path:

```console
pip install toeplitz-ls
```

The wheels are tagged `py3-none-<platform>`: the package is a ctypes binding, so
one wheel serves every Python version on that platform. Each wheel contains
several microarchitecture builds of the native library and loads the widest one
the CPU can execute, once, at import — see [Runtime dispatch](#runtime-dispatch).

Supported platforms: Linux on glibc and musl (`x86-64`, `aarch64`), Windows
(`x86-64`, `arm64`) and macOS (`x86-64`, `arm64`). The package requires
Python >= 3.8 and NumPy >= 1.17.3; the optional `tlsdd` extra requires
mpmath >= 1.0.

Building from source instead:

```console
git clone https://github.com/kkrutkowski/Toeplitz-LS.git --depth=1
cd ./Toeplitz-LS
make native        # also tunes the FFT fusion depth on this machine (about a minute)
                   # 'make TUNE=0 native' skips the tuning, 'make generic' is the
                   # portable baseline and skips it too
cd ./toeplitz-ls
pip3 install .
```

## Runtime dispatch

A wheel ships one shared object per microarchitecture level, plus a manifest
describing them:

| platform | variants |
|---|---|
| Linux x86-64 | `x86-64`, `x86-64-v2`, `x86-64-v2+avx`, `x86-64-v3`, `x86-64-v4` |
| Linux aarch64 | `armv8-a`, `armv8.2-a+sve-256`, `armv8.2-a+sve-512` |
| Windows x86-64 | same five x86 levels as Linux |
| Windows arm64 | same three ARM levels as Linux |
| macOS x86-64 | same five x86 levels as Linux |
| macOS arm64 | `armv8-a` (Apple silicon has no SVE, so there is nothing wider) |

`toeplitz_ls._dispatch` probes the CPU at import and loads the first variant
whose requirements are met, falling back to the baseline library. The features
come from the library itself (`tls_cpu_features()`, a CPUID/HWCAP probe in the
baseline `.so` that also checks the OS enabled the AVX state), not from an OS
summary such as `IsProcessorFeaturePresent` — so a variant is only ever chosen
when it can really execute. The choice is made once, so a periodogram call pays
nothing for it:

```python
import toeplitz_ls
toeplitz_ls.__dispatch__   # {'name': 'x86-64-v3', 'file': 'libtls-x86-64-v3.so', ...}
```

`TOEPLITZ_LS_DISPATCH` overrides the choice (`=list` prints what the wheel
contains and what this CPU supports; naming a variant this CPU cannot run is an
error rather than a crash). SVE libraries are only selected when the hardware
vector length matches the length they were compiled for; Linux reports it
through `prctl`, Windows through a `cntb` probe in the baseline library.

## FFT fusion depth

The FFT fuses several radix-2 stages into one pass, and the best depth depends
on the grid: short transforms like deeper passes (fewer twiddle loads), long
ones prefer shallower passes (less register pressure and a smaller working
set). `make native` and every release build therefore measure the depth per
power-of-two grid with `tests/bench_fuse --tune` and bake the result in as
`nanofft_depth_table.h`, which the library consults when the caller asks for
the automatic choice. This is a build artifact like the scaling calibration: it
is measured once per vector width on the build machine, and variants that cannot
run there borrow the measurement of the same vector width (the generated file
records which variant it came from).

`make TUNE=0 native` keeps the per-precision defaults instead, and
`toeplitz_ls` callers can still override the depth explicitly through
`tls_tuning_set_fft_fuse()`.

## Numerical accuracy of the built-in vector exponential

The library carries its own Remez implementation of `exp` for the SIMD paths
(`include/nanofft_vecmath.h`, coefficients from
`src/miscellaneous/remez_exp_lut.cpp`) instead of calling the platform's
vectorised exponential, because libmvec exists only in glibc and the same
sources would otherwise link differently on musl, macOS and Windows. It is
compiled with the caller's fast-math settings on purpose (the Horner chain
reassociation is a win), which costs a few ulp: `tests/check_vecmath` bounds it
relative to libm over the whole finite range, saturates to 0/inf where `exp`
does, and the double-double code is built without fast-math entirely. What
matters for the statistics — R² and NLL — is covered by
`tests/check_periodogram` and the Python test suite, whose tolerances are far
above these differences.

## Usage

The package exposes two periodogram families. The FastChi2/AoVMH modules (`tlsf`, `tls`, `tlsdd`) use the Toeplitz least-squares solve. The IHS modules (`ihsf`, `ihs`, `ihsdd`) use the incoherent harmonic sum: one NuFFT per harmonic, no linear solve, and only the `standard` or `nll` statistic. Each family has single-, double-, and double-double-precision variants. The single- and double-precision variants accept and return NumPy arrays. The double-double variants rely on compensated arithmetic and operate on a 106-bit mantissa; they use Mpmath's mpf type to bridge the gap between the library's internal representation and Python's native numeric types. The `get_peaks` helper can turn a periodogram into a sorted, interpolated peak list for printing or saving.

## Normalizations

For the FastChi2 modules (`tlsf`, `tls`, `tlsdd`), `power()` and `autopower()` accept `normalization=`:

| value | output | notes |
|---|---|---|
| `"standard"` (default) | coefficient of determination `R^2` | |
| `"anova"` | ANOVA / `F` statistic | Schwarzenberg-Czerny's ANOVA convention |
| `"nll"` | natural negative log false-alarm probability | `-ln(FAP)`|

For `nterms = d` harmonics:

- `"standard"` is the coefficient of determination `R^2` of the full model.
  Values at or above 1 are numerical artefacts of an ill-conditioned solve and
  are reported as 0.
- `"anova"` is the ANOVA `F` statistic in Schwarzenberg-Czerny's convention.
- `"nll"` is the natural-log negative false-alarm probability, `-ln(FAP)`,
  computed from the direct Pochhammer/incomplete-beta statistic with its
  finite-sample correction.

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
least-squares periodogram) but not orthogonal. In practice it is usually
slightly less accurate than FastChi2, but at least 3x less computationally
expensive: it evaluates `d` NuFFTs instead of `3d` and replaces the `O(d^2)`
normal-equation solve with an `O(d)` power accumulation.

The supported statistics are:

- `"standard"` (default): a direct estimate from the per-harmonic powers. For
  `nterms > 1` this is a sum, not a bounded `R^2`, and may exceed 1.
- `"nll"`: the IHS negative log false-alarm probability, using Wilkie's (1983)
  finite-sample correction for the Rayleigh Z test on each harmonic.

IHS is not suitable when `nterms^4 > M`; the Python wrappers emit a
`RuntimeWarning` in that case. Use the FastChi2 modules or reduce `nterms`.

## Performance

The hot path is a blocked NuFFT sweep: the frequency grid is cut into blocks,
each block is one transform of `2 x block` points, and the Toeplitz system of
every frequency in the block is solved immediately. Four things are tuned for
it (all measured with the interleaved harness in [`tests/`](tests/README.md)):

- **Fused-twiddle FFT.** A pass fuses `L` radix-2 DIF stages (a radix-`2^L`
  pass) so the buffer is traversed once per `L` stages instead of once per
  stage. `L` is a build-time constant, set separately per precision:

  | variable | precision | default |
  |---|---|---|
  | `NANOFFT_FFT_FUSE_FLOAT` | `tlsf` (float) | `floor(log2(VECF_LEN)) - 1` |
  | `NANOFFT_FFT_FUSE_DOUBLE` | `tls` (double) | `floor(log2(VEC_LEN))` |
  | `NANOFFT_FFT_FUSE_DD` | `tlsdd` (double-double) | 2 |

  i.e. 3 / 3 / 2 with 512-bit vectors, 2 / 2 / 2 with 256-bit, 1 / 1 / 2 with
  128-bit. The float pass takes the depth a double-precision vector of the same
  width would use: a fused pass keeps `2^(L+1)` data vectors plus ~6 twiddle
  and temporary registers live, so the full-lane depth fills the whole register
  file and measured never faster than one step shallower. The range is 1 (the
  classic stage-at-a-time transform) to 4 (radix-16). It is also a cap: the
  runtime switch `tls_tuning_set_fft_fuse()` (see `include/tls_tuning.h`,
  intended for benchmarking) can only select a depth the precision was built
  for. Build with e.g.

  ```console
  make native FFT_FUSE_FLOAT=4 FFT_FUSE_DOUBLE=2
  ```

  The leftover stages of an odd stage count go first (`TLS_FFT_ORDER_END`, the
  default: the first DIF pass streams the whole array and is bandwidth bound,
  the deepest fusion lands on the cache-resident tail), or last
  (`TLS_FFT_ORDER_FRONT`) via `tls_tuning_set_fft_order()`.

- **Vectorized COBRAVO bit reversal.** The COBRA tile reversal is emitted with
  AVX-512/AVX2 gathers from a contiguous staging buffer instead of the scalar
  in-tile swap, and the tile width shrinks with `log2(N)` so transforms below
  `2^13` use it too.

- **Deferred bit reversal.** The NuFFT can hand its block back in the FFT's
  bit-reversed *slot* order; the per-frequency sums, the Toeplitz solve and the
  normalization are all per-bin, so the results are permuted once per block
  instead of once per transform. This applies to every FastChi2 path (including
  the `nterms=1` GLS closed form) and to IHS with `nterms >= 2`. IHS at
  `nterms = 1` keeps the frequency-ordered path: it runs a single transform per
  block, so deferring the permutation cannot save anything.

- **Block-sized work arrays.** The trig-sum rows and the twiddle-ladder state
  are sized by the NuFFT block and the model order, not by the whole frequency
  grid, and the solver scratch is allocated once per periodogram instead of
  once per block.

Two knobs exist for benchmarking only: `tls_tuning_set_defer()` (0 restores
frequency-ordered output) and `tls_tuning_set_cobravo()` (0 restores the scalar
in-tile permutation). They are process-global and read when a plan is created;
the defaults above are what a plain build ships with.

Representative single-core numbers (AMD Ryzen 9 7945HX, gcc 14, `-O3
-ffast-math -march=native`), branch / unmodified `main`, best of 5 interleaved
rounds (`M` observations, `N` frequencies):

| workload | main | branch | ratio |
|---|---|---|---|
| `N=4096, nterms=3` | 524 us | 477 us | 0.91 |
| `N=8192, nterms=3` | 978 | 684 | 0.70 |
| `N=12288, nterms=3` | 1249 | 783 | 0.63 |
| `N=8192, nterms=1` | 258 | 237 | 0.92 |
| `N=8192, nterms=8` | 3005 | 1997 | 0.67 |
| `N=8192, nterms=3`, `nll` | 914 | 839 | 0.92 |
| `N=8192, nterms=3`, `pswf21` | 800 | 675 | 0.84 |
| `N=65536, nterms=3` | 5844 | 3161 | 0.54 |
| `N=131072, nterms=3` | 9200 | 5638 | 0.61 |
| IHS, `nterms=3` | 268 | 255 | 0.95 |
| IHS, `nterms=1` (unchanged path) | 125 | 125 | 1.00 |

## [Free-threaded (GIL-free) Python](https://docs.python.org/3/howto/free-threading-python.html) support
toeplitz-ls is compatible with free-threaded CPython (PEP 703, available as an optional build since Python 3.13). With the GIL disabled, ThreadPoolExecutor-based batch processing achieves true parallelism without the overhead of multiprocessing — no serialisation, no inter-process memory duplication. 

Note that Mpmath is not thread-safe by default; when using tlsdd in a multithreaded context, each worker should maintain its own mp context via mpmath.workprec() rather than mutating the global mp.prec.

## NaN handling
When the system matrix at a given frequency is numerically unstable, the solver returns NaN rather than raising an exception. Use np.nanargmax() and np.nanmax() in place of their standard counterparts to safely recover the peak frequency and power without short-circuiting on degenerate bins.

For manual filtering, call power() or autopower() with autonan=False to receive rough conditionality estimates alongside the periodogram. These estimates are solver-specific: LDLT reports a diagonal-ratio estimate, while the three Toeplitz solvers (levinson, zohar, and bareiss) return a reflection-coefficient-based upper bound `(2*nterms + 1) * P`. The Toeplitz estimates are generally much higher (and more reliable) for the same frequencies, so threshold values should be chosen with the solver family in mind.

### Example usage of tls and tlsf modules
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
    fmax=12, #default value
    nterms=1, # generalized Scargle periodogram
    oversampling=5, #frequency density, relative to a dense, uniform FFT
    backend="pswf43", # or "pswf21", "lra"
    normalization="standard" # or "anova", "nll"
    )

best_idx = np.nanargmax(r2)
best_freq = freq[best_idx]
best_power = r2[best_idx]

print(f"Peak frequency: {best_freq:.6g} (R^2 = {best_power:.6g})")

plt.figure(figsize=(12, 5))

plt.plot(freq, r2, label='degree = 1', color='red')
plt.axvline(best_freq, linestyle='--', color='black', alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Coefficient of determination')
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
    solver="levinson", # or 'zohar', 'bareiss', 'ldlt', 'svd'
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
make native        # build for this machine, tuning the FFT fusion depth
make -C tests check
make wheels        # every release wheel, each ABI-checked against its tag
make sdist         # source distribution; compiles on install for platforms without a wheel
```

Build artifacts (objects, staged wheel payloads, `.toolchains/`) live under
`.build/`.  Nothing is committed to it.
