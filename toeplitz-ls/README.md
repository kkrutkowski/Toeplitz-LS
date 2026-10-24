# toeplitz-ls

Installable Python interface for the Toeplitz least-squares FastChi2 shared
library built by the parent project.

## Build and install

From the repository root, build either shared-library variant:

```sh
make lib-generic
```

or:

```sh
make lib-native
```

Both targets build `lib/tls.so` and copy the same shared object into this
package as `toeplitz_ls/tls.so`. Then install from this directory:

```sh
cd toeplitz-ls
pip install .
```

NumPy is required. The double-double wrapper imports `mpmath` only when `tlsdd`
is used; install it with:

```sh
pip install ".[tlsdd]"
```

## Usage

```python
import numpy as np
from toeplitz_ls import ihs, tls, tlsf, tlsdd

t = np.linspace(0.0, 10.0, 128)
y = np.sin(2.0 * np.pi * 0.7 * t)

frequency, power = tls.autopower(t, y)
frequency_ihs, power_ihs = ihs.autopower(t, y, normalization="nll")
```

For the FastChi2 modules (`tlsf`, `tls`, `tlsdd`), `power()` and
`autopower()` accept `normalization=`:

- `"standard"` (default) - coefficient of determination `R^2`.
- `"anova"` - ANOVA / `F` statistic, following Schwarzenberg-Czerny's convention.
- `"nll"` - natural-log negative log false-alarm probability, `-ln(FAP)`;
  the exact Pochhammer/incomplete-beta statistic for FastChi2.
- `"regularized"` - Laplace-regularized Cybenko statistic, matching
  ihsnpeaks' `bayes` (Relative Evidence Ratio). For `nterms > 1` it requires
  a Toeplitz solver (`levinson`, `zohar`, or `bareiss`).

The `cond` array returned with `autonan=False` is the condition upper bound
`(2*nterms + 1) * P`, where `P` is Cybenko's reflection-coefficient product;
`"regularized"` uses `P`, not the bound. See the parent README for the
formulas.

The IHS modules (`ihsf`, `ihs`, `ihsdd`) are a simpler alternative that uses
one NuFFT per harmonic and no linear solve. They expose the same `power()` /
`autopower()` shape but support only:

- `"standard"` (default) - sum of the uncorrelated per-harmonic `R^2`
  contributions.
- `"nll"` - the IHS negative log false-alarm probability, using Wilkie's
  finite-sample correction on each harmonic followed by the Erlang
  logarithmic correction.

They do not accept `solver=` or `autonan=` and do not return a condition
array. Under uniformly sampled observation times their statistical efficiency
is approximately `2/pi` of the full chi-square periodogram.

The shared library is loaded with `ctypes.CDLL`, so CPython releases the GIL
during the C calls. Independent `power()` or `autopower()` calls can therefore
run concurrently from a `concurrent.futures.ThreadPoolExecutor` while the
native calculation is executing.
