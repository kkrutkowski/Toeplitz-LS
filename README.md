# [WIP] Toeplitz-LS
Reference implementation of multiharmonic least-squares periodogram (AoVMH(W) / FastChi²) using blocked NuFFT and a fast Toeplitz solver for complex coefficients.

## Installation
The codebase is currently in a pre-release state (versioned as 0.0.0), so breaking changes may occur and no release builds are available yet. As such compilation from source is recommended.

To install the Python wrapper:
```console
git clone https://github.com/kkrutkowski/Toeplitz-ls.git --depth=1
cd ./Toeplitz_ls
make native  # use 'make generic' if build time is a priority — this may take a few minutes
cd ./toeplitz-ls
pip3 install .
```

## Usage

The toeplitz-ls package exposes three submodules, each targeting a different precision level. The single-precision (tlsf) and double-precision (tls) submodules accept and return NumPy arrays. The double-double module (tlsdd) relies on compensated arithmetic and operates on a 106-bit mantissa; it uses Mpmath's mpf type to bridge the gap between the library's internal representation and Python's native numeric types.

## [Free-threaded (GIL-free) Python](https://docs.python.org/3/howto/free-threading-python.html) support
toeplitz-ls is compatible with free-threaded CPython (PEP 703, available as an optional build since Python 3.13). With the GIL disabled, ThreadPoolExecutor-based batch processing achieves true parallelism without the overhead of multiprocessing — no serialisation, no inter-process memory duplication. 

Note that Mpmath is not thread-safe by default; when using tlsdd in a multithreaded context, each worker should maintain its own mp context via mpmath.workprec() rather than mutating the global mp.prec.

## NaN handling
When the system matrix at a given frequency is numerically unstable, the solver returns NaN rather than raising an exception. Use np.nanargmax() and np.nanmax() in place of their standard counterparts to safely recover the peak frequency and power without short-circuiting on degenerate bins.

For manual filtering, call power() or autopower() with autonan=False to receive rough conditionality estimates alongside the periodogram. These estimates are solver-specific: LDLT reports a diagonal-ratio estimate, while the three Toeplitz solvers (levinson, zohar, and bareiss) return a reflection-coefficient-based upper bound. The Toeplitz estimates are generally much higher (and more reliable) for the same frequencies, so threshold values should be chosen with the solver family in mind.

### Example usage of tls and tlsf modules
```python
import numpy as np
import matplotlib.pyplot as plt
from toeplitz_ls import tlsf

rng = np.random.default_rng(seed=123)
N = 1000
t = np.sort(rng.uniform(0, 1000, size=N))
y = np.sin(50 * t) + 0.5 * np.sin(150 * t) + 1 + rng.poisson(size=N)

# Compute default periodogram
freq, r2 = tlsf.autopower(
    t, y, dy=None,
    fmax=12, #default value
    nterms=3, #default number of terms
    oversampling=5, #frequency density, relative to a dense, uniform FFT
    solver="levinson", # or "bareiss", "zohar", "ldlt"
    backend="pswf", # or "lra"
    normalization="standard" # or "asymptotic"
    )

best_idx = np.nanargmax(r2)
best_freq = freq[best_idx]
best_power = r2[best_idx]

print(f"Peak frequency: {best_freq:.6g} (R^2 = {best_power:.6g})")

plt.figure(figsize=(12, 5))

plt.plot(freq, r2, label='degree = 3', color='red')
plt.axvline(best_freq, linestyle='--', color='black', alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Coefficient of determination')
plt.legend()

plt.tight_layout()
plt.show()
```


### Example usage with a custom singularity threshold
```python
import numpy as np
import matplotlib.pyplot as plt
from toeplitz_ls import tls

rng = np.random.default_rng(seed=123)
N = 1000

t = np.sort(rng.uniform(0, 1000, size=N))
y = np.sin(50 * t) + 0.5 * np.sin(150 * t) + 1 + rng.poisson(size=N)

# Ask for the raw power and LDLT conditionality estimates.
freq, power, cond = tls.autopower(
    t, y, dy=None,
    fmax=20,
    nterms=1, # mathematically equivalent to the generalized Scargle periodogram
    oversampling=5,
    solver="ldlt", # LDLT condition estimates are on a different scale than Toeplitz bounds.
    backend="pswf", # or "lra"
    normalization="asymptotic",
    autonan=False
)

threshold = 1e2

# Mask numerically unreliable estimates so the peak search remains consistent.
power[cond > threshold] = np.nan

best_idx = np.nanargmax(power)
best_freq = freq[best_idx]
best_power = power[best_idx]

print(f"Peak frequency: {best_freq:.6g} (R^2 = {best_power:.6g})")

plt.figure(figsize=(12, 5))
plt.plot(freq, power, label='degree = 1', color='blue')
plt.axvline(best_freq, linestyle='--', color='black', alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Negative log-likelihood')
plt.legend()
plt.tight_layout()
plt.show()
```
