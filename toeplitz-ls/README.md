# toeplitz-ls

Installable Python interface for the Toeplitz least-squares FastChi2 shared
library built by the parent project.

## Build and install

From the repository root, build either shared-library variant:

```sh
make generic
```

or:

```sh
make native
```

Both targets build `lib/tls.so` and copy the same shared object into this
package as `toeplitz_ls/tls.so`. Then install from this directory:

```sh
cd toeplitz-ls
pip install .
```

NumPy >= 1.17.3 is required. The double-double wrapper imports `mpmath` only
when `tlsdd` is used; install mpmath >= 1.0 with:

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
frequency_ihs, power_ihs = ihs.autopower(t, y, normalization="corrected")
```

For the FastChi2 modules (`tlsf`, `tls`, `tlsdd`), `power()` and
`autopower()` accept `normalization=`:

- `"standard"` (default) - coefficient of determination `R^2` of the full
  model. Values at or above 1 are numerical artefacts of an ill-conditioned
  solve and are reported as 0.
- `"anova"` - ANOVA / `F` statistic, following Schwarzenberg-Czerny's convention.
- `"nll"` - natural-log negative log false-alarm probability, `-ln(FAP)`;
  the direct Pochhammer/incomplete-beta statistic with its finite-sample
  correction. `Neff = (sum w)^2 / sum(w^2)` with `w = 1/dy^2`.

The `cond` array returned with `autonan=False` is the condition upper bound
`(2*nterms + 1) * P`, where `P` is Cybenko's reflection-coefficient product.

The IHS modules (`ihsf`, `ihs`, `ihsdd`) are a simplified formulation of a
similar periodogram. They assume that the harmonic frequencies are independent
(a regression from the standard least-squares periodogram) but not orthogonal,
use one NuFFT per harmonic and no linear solve, and expose the same `power()` /
`autopower()` shape. They are usually slightly less accurate than FastChi2, but
at least 3x less computationally expensive. They support only:

- `"raw"` - a direct display statistic from the normalized per-harmonic
  Rayleigh powers. It is not a calibrated false-alarm probability, and for
  `nterms > 1` it is not a bounded `R^2`; it may exceed 1.
- `"corrected"` - the IHS negative log false-alarm probability. Each harmonic's
  Rayleigh power gets Wilkie's (1983) finite-sample correction, then the
  harmonic sum gets the Erlang correction. `Neff = (sum |wy|)^2 / sum(wy^2)`,
  where `wy = w * (y - ymean)`. Use this for significance estimates.
- `"default"` (the default) - the initial formula and Erlang correction without
  the finite-sample Wilkie correction.

They do not accept `solver=` or `autonan=` and do not return a condition
array. IHS is not suitable when `nterms^4 > Neff`, where `Neff` is the
raw-weight effective sample size; the wrappers emit a `RuntimeWarning` in that
case.

The shared library is written almost exclusively in C and loaded with
`ctypes.CDLL`, so independent `power()`/`autopower()` calls release the GIL and
can run in parallel from a `concurrent.futures.ThreadPoolExecutor`.
Multithreading is expected to give major performance gains even on regular
CPython; for optimum performance, use a free-threaded build (PEP 703).
