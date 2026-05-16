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
When the system matrix at a given frequency is numerically singular, the solver returns NaN rather than raising an exception. Use np.nanargmax() and np.nanmax() in place of their standard counterparts to safely recover the peak frequency and power without short-circuiting on degenerate bins.

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
    solver="levinson", # or "bareiss"
    # supports "zohar" and "ldlt" solvers as well, although these options are not recommended
    backend="pswf", # or "lra"
    normalization="standard" # or "asymptotic"
    )

plt.figure(figsize=(12, 5))

plt.plot(freq, r2, label='degree = 3', color='red')
plt.xlabel('Frequency')
plt.ylabel('Coefficient of determination')
plt.legend()

plt.tight_layout()
plt.show()
```


### Example usage of tlsdd module
```python
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, sin, matrix
from toeplitz_ls import tlsdd

# Set working precision to 106-bit mantissa (double-double equivalent)
mp.prec = 106

rng = np.random.default_rng(seed=123)
N = 1000

t = sorted([mpf(v) for v in rng.uniform(0, 1000, size=N)])
noise = [mpf(v) for v in rng.poisson(size=N)]
y = [sin(mpf(50) * t[i]) + mpf("0.5") * sin(mpf(25) * t[i]) + mpf(1) + noise[i] for i in range(N)]

t_arr = np.array(t, dtype=object)
y_arr = np.array(y, dtype=object)

# Compute the periodogram
freq, nll = tlsdd.autopower(
    t_arr, y_arr, dy=None,
    fmax=20,
    nterms=1,               # equivalent to Generalised Scargle Periodogram
    oversampling=5,
    solver="bareiss",          # supports "zohar" and "ldlt" as well, although these options are not recommended
    backend="pswf",         # or "lra"
    normalization="asymptotic" # asymptotic estimate of negative log-likelihood
)

# Use nanargmax/nanmax — the solver returns NaN at numerically singular frequencies
best_idx = np.nanargmax(nll)
best_freq = freq[best_idx]
best_power = np.nanmax(nll)

plt.figure(figsize=(12, 5))
plt.plot(freq, nll, label='nterms = 1', color='blue')
plt.xlabel('Frequency')
plt.ylabel('−log L (asymptotic)')
plt.legend()
plt.tight_layout()
plt.show()
```
