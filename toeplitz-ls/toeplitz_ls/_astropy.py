"""Astropy ``LombScargle`` backend registration for ``toeplitz_ls``.

Exposes the CPU FastChi2 wrappers ``'tls'`` (double precision), ``'tlsf'``
(single precision), and ``'tlsh'`` (half precision / IEEE binary16) as
``method`` options for :class:`astropy.timeseries.LombScargle`.

Registration is lazy: if Astropy's Lomb-Scargle module is already imported
when :mod:`toeplitz_ls` is imported, it is patched immediately; otherwise a
lightweight :data:`sys.meta_path` finder hooks the module the first time
Astropy's Lomb-Scargle implementation is imported, adding zero startup
overhead when Astropy is unused or not installed.
"""

from __future__ import annotations

import functools
import importlib.abc
import math
import sys
import threading

import numpy as np

from ._fastchi2 import tls, tlsf, tlsh

_TARGET_MAIN = "astropy.timeseries.periodograms.lombscargle.implementations.main"
_TARGET_IMPL = "astropy.timeseries.periodograms.lombscargle.implementations"
_TARGET_CORE = "astropy.timeseries.periodograms.lombscargle.core"

_ASTROPY_NORMALIZATIONS = ("standard", "model", "log", "psd")
_TLS_EXTRA_NORMALIZATIONS = ("anova", "nll")
_ALLOWED_NORMALIZATIONS = _ASTROPY_NORMALIZATIONS + _TLS_EXTRA_NORMALIZATIONS
_ALLOWED_METHOD_KWDS = frozenset({"backend", "solver", "spreading", "vectorize"})
_METHOD_NAMES = ("tls", "tlsf", "tlsh")

_PATCH_LOCK = threading.Lock()


def _prepare_inputs(t, y, dy):
    if dy is None:
        t_arr, y_arr = np.broadcast_arrays(t, y)
        t_arr = np.ascontiguousarray(t_arr, dtype=np.float64)
        y_arr = np.ascontiguousarray(y_arr, dtype=np.float64)
        dy_arr = None
    else:
        t_arr, y_arr, dy_arr = np.broadcast_arrays(t, y, dy)
        t_arr = np.ascontiguousarray(t_arr, dtype=np.float64)
        y_arr = np.ascontiguousarray(y_arr, dtype=np.float64)
        dy_arr = np.ascontiguousarray(dy_arr, dtype=np.float64)

    if t_arr.ndim != 1:
        raise ValueError("t, y, dy should be one dimensional")
    return t_arr, y_arr, dy_arr


def _chi2_ref(y_arr, dy_arr):
    if dy_arr is None:
        centered = y_arr - float(np.mean(y_arr))
        return float(np.dot(centered, centered))
    w = 1.0 / (dy_arr * dy_arr)
    y_mean = float(np.dot(w, y_arr) / np.sum(w))
    yw = (y_arr - y_mean) / dy_arr
    return float(np.dot(yw, yw))


def _make_backend(mod, min_nf_default, min_nf_lra):
    def _lombscargle_backend(
        t,
        y,
        dy,
        f0,
        df,
        Nf,
        center_data=True,
        fit_mean=True,
        normalization="standard",
        nterms=1,
        **method_kwds,
    ):
        del center_data  # Identical to center_data=True when fit_mean=True.
        if not fit_mean:
            raise ValueError("toeplitz_ls Astropy backend requires fit_mean=True")

        if not isinstance(normalization, str):
            raise ValueError(f"normalization={normalization!r} not recognized")
        norm = normalization.strip().lower()
        if norm not in _ALLOWED_NORMALIZATIONS:
            raise ValueError(f"normalization='{normalization}' not recognized")

        unexpected = set(method_kwds) - _ALLOWED_METHOD_KWDS
        if unexpected:
            bad = ", ".join(sorted(unexpected))
            raise TypeError(f"unexpected method_kwds for toeplitz_ls Astropy backend: {bad}")

        t_arr, y_arr, dy_arr = _prepare_inputs(t, y, dy)

        if Nf <= 0:
            raise ValueError("Number of frequencies must be positive")
        f0_val = float(f0)
        df_val = float(df)
        if f0_val < 0.0:
            raise ValueError("Frequencies must be positive")
        if Nf > 1 and df_val <= 0.0:
            raise ValueError("Frequency steps must be positive")
        if Nf == 1 and (not math.isfinite(df_val) or df_val <= 0.0):
            df_val = 1.0

        backend_name = method_kwds.get("backend", "pswf43")
        is_lra = isinstance(backend_name, str) and backend_name.strip().lower() == "lra"
        min_nf = min_nf_lra if is_lra else min_nf_default
        eval_nf = max(int(Nf), min_nf)

        tls_norm = norm if norm in ("standard", "anova", "nll") else "standard"
        raw = mod.power(
            eval_nf,
            df_val,
            f0_val,
            t_arr,
            y_arr,
            dy_arr,
            nterms=nterms,
            normalization=tls_norm,
            autonan=True,
            **method_kwds,
        )

        power = np.array(raw[:Nf], dtype=np.float64, copy=True)
        power[np.isnan(power)] = 0.0

        if norm == "model":
            power = power / (1.0 - power)
        elif norm == "log":
            power = -np.log1p(-power)
        elif norm == "psd":
            power = 0.5 * _chi2_ref(y_arr, dy_arr) * power

        return power

    return _lombscargle_backend


_ASTROPY_METHODS = {
    "tls": _make_backend(tls, min_nf_default=8, min_nf_lra=8),
    "tlsf": _make_backend(tlsf, min_nf_default=16, min_nf_lra=32),
    "tlsh": _make_backend(tlsh, min_nf_default=16, min_nf_lra=32),
}


def _patch_astropy_main(main_mod):
    with _PATCH_LOCK:
        if not getattr(main_mod, "_toeplitz_ls_patched", False):
            main_mod.METHODS.update(_ASTROPY_METHODS)

            orig_available = main_mod.available_methods

            @functools.wraps(orig_available)
            def _available_methods():
                methods = list(orig_available())
                for name in _METHOD_NAMES:
                    if name not in methods:
                        methods.append(name)
                return methods

            orig_lombscargle = main_mod.lombscargle

            @functools.wraps(orig_lombscargle)
            def _lombscargle(
                t,
                y,
                dy=None,
                frequency=None,
                method="auto",
                assume_regular_frequency=False,
                normalization="standard",
                fit_mean=True,
                center_data=True,
                method_kwds=None,
                nterms=1,
            ):
                if method in _ASTROPY_METHODS:
                    freq = np.asarray(frequency)
                    output_shape = freq.shape
                    freq_1d = freq.ravel()
                    f0, df, Nf = main_mod._get_frequency_grid(freq_1d, assume_regular_frequency)
                    kwds = dict(method_kwds or {})
                    power = _ASTROPY_METHODS[method](
                        t,
                        y,
                        dy,
                        f0,
                        df,
                        Nf,
                        center_data=center_data,
                        fit_mean=fit_mean,
                        normalization=normalization,
                        nterms=nterms,
                        **kwds,
                    )
                    return power.reshape(output_shape)
                return orig_lombscargle(
                    t,
                    y,
                    dy=dy,
                    frequency=frequency,
                    method=method,
                    assume_regular_frequency=assume_regular_frequency,
                    normalization=normalization,
                    fit_mean=fit_mean,
                    center_data=center_data,
                    method_kwds=method_kwds,
                    nterms=nterms,
                )

            main_mod.available_methods = _available_methods
            main_mod.lombscargle = _lombscargle
            main_mod._toeplitz_ls_patched = True

        impl_mod = sys.modules.get(_TARGET_IMPL)
        if impl_mod is not None:
            impl_mod.available_methods = main_mod.available_methods
            impl_mod.lombscargle = main_mod.lombscargle

        core_mod = sys.modules.get(_TARGET_CORE)
        if core_mod is not None:
            core_mod.available_methods = main_mod.available_methods
            core_mod.lombscargle = main_mod.lombscargle
            if hasattr(core_mod, "LombScargle"):
                core_mod.LombScargle.available_methods = main_mod.available_methods()


def _remove_hook_finder(finder):
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass


class _AstropyPostExecLoader(importlib.abc.Loader):
    def __init__(self, wrapped, finder):
        self._wrapped = wrapped
        self._finder = finder

    def create_module(self, spec):
        if hasattr(self._wrapped, "create_module"):
            return self._wrapped.create_module(spec)
        return None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _patch_astropy_main(module)
        _remove_hook_finder(self._finder)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _AstropyHookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MAIN:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _AstropyPostExecLoader(spec.loader, self)
                return spec
        return None


def register_astropy_methods():
    """Register ``'tls'``, ``'tlsf'``, and ``'tlsh'`` with Astropy lazily."""
    main_mod = sys.modules.get(_TARGET_MAIN)
    if main_mod is not None:
        _patch_astropy_main(main_mod)
        return

    with _PATCH_LOCK:
        main_mod = sys.modules.get(_TARGET_MAIN)
        if main_mod is not None:
            _patch_astropy_main(main_mod)
            return
        if not any(isinstance(f, _AstropyHookFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _AstropyHookFinder())


__all__ = ["register_astropy_methods"]
