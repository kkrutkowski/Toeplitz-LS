"""Shared ctypes ABI helpers and wrapper factories for GPU offload backends (SYCL, Vulkan)."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .._fastchi2 import (
    _auto_grid,
    _check_grid,
    _check_int,
    _check_normalization,
    _check_observation_count,
    _prepare_half_inputs,
    _prepare_numpy_inputs,
)
from .._ihs import _check_ihs_normalization, _raw_neff, _warn_if_ihs_undersampled

OFFLOAD_BACKENDS = {"pswf43": 1, "pswf21": 2}
_DBL_PTR = ctypes.POINTER(ctypes.c_double)
_FLT_PTR = ctypes.POINTER(ctypes.c_float)
_NULL_DBL_PTR = _DBL_PTR()
_NULL_FLT_PTR = _FLT_PTR()


class CDeviceInfo(ctypes.Structure):
    """Shared C ABI device descriptor layout for tls_sycl.so and tls_vulkan.so."""

    _fields_ = [
        ("id", ctypes.c_int),
        ("name", ctypes.c_char * 256),
        ("backend", ctypes.c_char * 32),
        ("device_type", ctypes.c_char * 32),
        ("max_compute_units", ctypes.c_uint64),
        ("global_mem_size", ctypes.c_uint64),
        ("is_gpu", ctypes.c_int),
        ("has_fp64", ctypes.c_int),
        ("has_fp16", ctypes.c_int),
    ]


@dataclass(frozen=True)
class OffloadDevice:
    """Base metadata descriptor for an available hardware offload target."""

    id: int
    name: str
    backend: str
    device_type: str
    max_compute_units: int
    global_mem_size: int
    is_gpu: bool
    has_fp64: bool
    has_fp16: bool

    def __str__(self) -> str:
        gib = self.global_mem_size / (1024.0**3) if self.global_mem_size else 0.0
        return (
            f"[{self.id}] {self.name} ({self.backend}/{self.device_type}, "
            f"{self.max_compute_units} CUs, {gib:.2f} GiB, "
            f"fp16={'yes' if self.has_fp16 else 'no'}, "
            f"fp64={'yes' if self.has_fp64 else 'no'})"
        )


def bind_offload_abi(lib, prefix: str) -> None:
    """Declare ctypes argtypes/restype for all ``<prefix>_*`` offload C ABI functions."""
    getattr(lib, f"{prefix}_abi_version").argtypes = []
    getattr(lib, f"{prefix}_abi_version").restype = ctypes.c_int

    getattr(lib, f"{prefix}_device_count").argtypes = []
    getattr(lib, f"{prefix}_device_count").restype = ctypes.c_int

    getattr(lib, f"{prefix}_device_info").argtypes = [ctypes.c_int, ctypes.POINTER(CDeviceInfo)]
    getattr(lib, f"{prefix}_device_info").restype = ctypes.c_int

    getattr(lib, f"{prefix}_last_error").argtypes = []
    getattr(lib, f"{prefix}_last_error").restype = ctypes.c_char_p

    dbl_p = ctypes.POINTER(ctypes.c_double)
    flt_p = ctypes.POINTER(ctypes.c_float)

    def _bind_fastchi2(name: str, y_p, out_p) -> None:
        fn = getattr(lib, name)
        fn.argtypes = [
            dbl_p,  # t
            y_p,  # y
            y_p,  # dy
            ctypes.c_int,  # M
            ctypes.c_double,  # f0
            ctypes.c_double,  # df
            ctypes.c_int,  # N
            ctypes.c_int,  # degree
            ctypes.c_int,  # backend
            ctypes.c_int,  # normalization
            ctypes.c_int,  # device_id
            out_p,  # power
            out_p,  # cond
        ]
        fn.restype = ctypes.c_int

    def _bind_ihs(name: str, y_p, out_p) -> None:
        fn = getattr(lib, name)
        fn.argtypes = [
            dbl_p,  # t
            y_p,  # y
            y_p,  # dy
            ctypes.c_int,  # M
            ctypes.c_double,  # f0
            ctypes.c_double,  # df
            ctypes.c_int,  # N
            ctypes.c_int,  # degree
            ctypes.c_int,  # backend
            ctypes.c_int,  # normalization
            ctypes.c_int,  # device_id
            out_p,  # power
        ]
        fn.restype = ctypes.c_int

    _bind_fastchi2(f"{prefix}_h_fastchi2", flt_p, flt_p)
    _bind_fastchi2(f"{prefix}_h_fastchi2_d", dbl_p, flt_p)
    _bind_fastchi2(f"{prefix}_f_fastchi2", flt_p, flt_p)
    _bind_fastchi2(f"{prefix}_df_fastchi2", dbl_p, dbl_p)
    _bind_fastchi2(f"{prefix}_d_fastchi2", dbl_p, dbl_p)

    _bind_ihs(f"{prefix}_h_ihs", flt_p, flt_p)
    _bind_ihs(f"{prefix}_h_ihs_d", dbl_p, flt_p)
    _bind_ihs(f"{prefix}_f_ihs", flt_p, flt_p)
    _bind_ihs(f"{prefix}_df_ihs", dbl_p, dbl_p)
    _bind_ihs(f"{prefix}_d_ihs", dbl_p, dbl_p)


def check_offload_backend_id(backend, label: str) -> int:
    if isinstance(backend, str):
        key = backend.strip().lower()
        if key in OFFLOAD_BACKENDS:
            return OFFLOAD_BACKENDS[key]
    raise ValueError(f"{label} offload backend must be 'pswf43' or 'pswf21'")


def check_device_id(device) -> int:
    return _check_int("device", device, 0)


def make_fastchi2_classes(
    *,
    label: str,
    prefix: str,
    module_name: str,
    load_library,
    raise_status,
):
    """Create ``(tlsh, tlsf, tlsdf, tls)`` wrapper classes bound to an offload C ABI."""
    fn_h = f"{prefix}_h_fastchi2"
    fn_hd = f"{prefix}_h_fastchi2_d"
    fn_f = f"{prefix}_f_fastchi2"
    fn_df = f"{prefix}_df_fastchi2"
    fn_d = f"{prefix}_d_fastchi2"
    _fn_cache: dict[str, object] = {}

    def _get_fn(name: str):
        fn = _fn_cache.get(name)
        if fn is None:
            fn = getattr(load_library(), name)
            _fn_cache[name] = fn
        return fn

    def _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan):
        out = np.empty(Nf, dtype=np.float32)
        cond = None if autonan else np.empty(Nf, dtype=np.float32)
        cond_ptr = _NULL_FLT_PTR if cond is None else cond.ctypes.data_as(_FLT_PTR)
        if use_double:
            status = _get_fn(fn_hd)(
                t_arr.ctypes.data_as(_DBL_PTR),
                y_arr.ctypes.data_as(_DBL_PTR),
                dy_arr.ctypes.data_as(_DBL_PTR),
                t_arr.size,
                f0,
                df,
                Nf,
                nterms,
                backend_id,
                normalization,
                device,
                out.ctypes.data_as(_FLT_PTR),
                cond_ptr,
            )
            raise_status(fn_hd, status)
        else:
            status = _get_fn(fn_h)(
                t_arr.ctypes.data_as(_DBL_PTR),
                y_arr.ctypes.data_as(_FLT_PTR),
                dy_arr.ctypes.data_as(_FLT_PTR),
                t_arr.size,
                f0,
                df,
                Nf,
                nterms,
                backend_id,
                normalization,
                device,
                out.ctypes.data_as(_FLT_PTR),
                cond_ptr,
            )
            raise_status(fn_h, status)
        if autonan:
            return out
        return out, cond

    def _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan):
        out = np.empty(Nf, dtype=np.float32)
        cond = None if autonan else np.empty(Nf, dtype=np.float32)
        cond_ptr = _NULL_FLT_PTR if cond is None else cond.ctypes.data_as(_FLT_PTR)
        status = _get_fn(fn_f)(
            t_arr.ctypes.data_as(_DBL_PTR),
            y_arr.ctypes.data_as(_FLT_PTR),
            dy_arr.ctypes.data_as(_FLT_PTR),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(_FLT_PTR),
            cond_ptr,
        )
        raise_status(fn_f, status)
        if autonan:
            return out
        return out, cond

    def _run_double_prepared(fn_name, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan):
        out = np.empty(Nf, dtype=np.float64)
        cond = None if autonan else np.empty(Nf, dtype=np.float64)
        cond_ptr = _NULL_DBL_PTR if cond is None else cond.ctypes.data_as(_DBL_PTR)
        status = _get_fn(fn_name)(
            t_arr.ctypes.data_as(_DBL_PTR),
            y_arr.ctypes.data_as(_DBL_PTR),
            dy_arr.ctypes.data_as(_DBL_PTR),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(_DBL_PTR),
            cond_ptr,
        )
        raise_status(fn_name, status)
        if autonan:
            return out
        return out, cond

    class tlsh:
        __module__ = module_name
        __doc__ = f"Half-precision (IEEE binary16) {label} GPU FastChi2 periodogram wrapper."

        @staticmethod
        def power(
            device,
            Nf,
            df,
            f0,
            t,
            y,
            dy=None,
            backend="pswf43",
            nterms=3,
            normalization="standard",
            *,
            autonan=True,
        ):
            """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(t, y, dy)
            _check_observation_count(t_arr.size, nterms)
            return _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan)

        @staticmethod
        def autopower(
            device,
            x,
            y,
            dy=None,
            fmin=None,
            fmax=12,
            oversampling=5,
            normalization="standard",
            nterms=3,
            backend="pswf43",
            *,
            autonan=True,
        ):
            """Return automatic frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(x, y, dy)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization, autonan)
            if autonan:
                return frequency, result
            power, cond = result
            return frequency, power, cond

    class tlsf:
        __module__ = module_name
        __doc__ = f"Single-precision (FP32) {label} GPU FastChi2 periodogram wrapper."

        @staticmethod
        def power(
            device,
            Nf,
            df,
            f0,
            t,
            y,
            dy=None,
            backend="pswf43",
            nterms=3,
            normalization="standard",
            *,
            autonan=True,
        ):
            """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
            _check_observation_count(t_arr.size, nterms)
            return _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)

        @staticmethod
        def autopower(
            device,
            x,
            y,
            dy=None,
            fmin=None,
            fmax=12,
            oversampling=5,
            normalization="standard",
            nterms=3,
            backend="pswf43",
            *,
            autonan=True,
        ):
            """Return automatic frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)
            if autonan:
                return frequency, result
            power, cond = result
            return frequency, power, cond

    class tlsdf:
        __module__ = module_name
        __doc__ = f"Dekker double-float (2xFP32) {label} GPU FastChi2 periodogram wrapper (float64 I/O)."

        @staticmethod
        def power(
            device,
            Nf,
            df,
            f0,
            t,
            y,
            dy=None,
            backend="pswf43",
            nterms=3,
            normalization="standard",
            *,
            autonan=True,
        ):
            """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            return _run_double_prepared(fn_df, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)

        @staticmethod
        def autopower(
            device,
            x,
            y,
            dy=None,
            fmin=None,
            fmax=12,
            oversampling=5,
            normalization="standard",
            nterms=3,
            backend="pswf43",
            *,
            autonan=True,
        ):
            """Return automatic frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_double_prepared(fn_df, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)
            if autonan:
                return frequency, result
            power, cond = result
            return frequency, power, cond

    class tls:
        __module__ = module_name
        __doc__ = f"Native double-precision (FP64) {label} GPU FastChi2 periodogram wrapper."

        @staticmethod
        def power(
            device,
            Nf,
            df,
            f0,
            t,
            y,
            dy=None,
            backend="pswf43",
            nterms=3,
            normalization="standard",
            *,
            autonan=True,
        ):
            """Return periodogram power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            return _run_double_prepared(fn_d, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)

        @staticmethod
        def autopower(
            device,
            x,
            y,
            dy=None,
            fmin=None,
            fmax=12,
            oversampling=5,
            normalization="standard",
            nterms=3,
            backend="pswf43",
            *,
            autonan=True,
        ):
            """Return automatic frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_double_prepared(fn_d, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization, autonan)
            if autonan:
                return frequency, result
            power, cond = result
            return frequency, power, cond

    return tlsh, tlsf, tlsdf, tls


def make_ihs_classes(
    *,
    label: str,
    prefix: str,
    module_name: str,
    load_library,
    raise_status,
):
    """Create ``(ihsh, ihsf, ihsdf, ihs)`` wrapper classes bound to an offload C ABI."""
    fn_h = f"{prefix}_h_ihs"
    fn_hd = f"{prefix}_h_ihs_d"
    fn_f = f"{prefix}_f_ihs"
    fn_df = f"{prefix}_df_ihs"
    fn_d = f"{prefix}_d_ihs"
    _fn_cache: dict[str, object] = {}

    def _get_fn(name: str):
        fn = _fn_cache.get(name)
        if fn is None:
            fn = getattr(load_library(), name)
            _fn_cache[name] = fn
        return fn

    def _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization):
        _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
        out = np.empty(Nf, dtype=np.float32)
        if use_double:
            status = _get_fn(fn_hd)(
                t_arr.ctypes.data_as(_DBL_PTR),
                y_arr.ctypes.data_as(_DBL_PTR),
                dy_arr.ctypes.data_as(_DBL_PTR),
                t_arr.size,
                f0,
                df,
                Nf,
                nterms,
                backend_id,
                normalization,
                device,
                out.ctypes.data_as(_FLT_PTR),
            )
            raise_status(fn_hd, status)
        else:
            status = _get_fn(fn_h)(
                t_arr.ctypes.data_as(_DBL_PTR),
                y_arr.ctypes.data_as(_FLT_PTR),
                dy_arr.ctypes.data_as(_FLT_PTR),
                t_arr.size,
                f0,
                df,
                Nf,
                nterms,
                backend_id,
                normalization,
                device,
                out.ctypes.data_as(_FLT_PTR),
            )
            raise_status(fn_h, status)
        return out

    def _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization):
        _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
        out = np.empty(Nf, dtype=np.float32)
        status = _get_fn(fn_f)(
            t_arr.ctypes.data_as(_DBL_PTR),
            y_arr.ctypes.data_as(_FLT_PTR),
            dy_arr.ctypes.data_as(_FLT_PTR),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(_FLT_PTR),
        )
        raise_status(fn_f, status)
        return out

    def _run_ihs_double_prepared(fn_name, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization):
        _warn_if_ihs_undersampled(_raw_neff(dy_arr), nterms)
        out = np.empty(Nf, dtype=np.float64)
        status = _get_fn(fn_name)(
            t_arr.ctypes.data_as(_DBL_PTR),
            y_arr.ctypes.data_as(_DBL_PTR),
            dy_arr.ctypes.data_as(_DBL_PTR),
            t_arr.size,
            f0,
            df,
            Nf,
            nterms,
            backend_id,
            normalization,
            device,
            out.ctypes.data_as(_DBL_PTR),
        )
        raise_status(fn_name, status)
        return out

    class ihsh:
        __module__ = module_name
        __doc__ = f"Half-precision (IEEE binary16) {label} GPU IHS periodogram wrapper."

        @staticmethod
        def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
            """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(t, y, dy)
            _check_observation_count(t_arr.size, nterms)
            return _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization)

        @staticmethod
        def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
            """Return automatic half-precision IHS frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr, use_double = _prepare_half_inputs(x, y, dy)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_ihs_half_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, use_double, backend_id, nterms, normalization)
            return frequency, result

    class ihsf:
        __module__ = module_name
        __doc__ = f"Single-precision (FP32) {label} GPU IHS periodogram wrapper."

        @staticmethod
        def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
            """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float32)
            _check_observation_count(t_arr.size, nterms)
            return _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

        @staticmethod
        def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
            """Return automatic single-precision IHS frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float32)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=16)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_ihs_float_prepared(device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
            return frequency, result

    class ihsdf:
        __module__ = module_name
        __doc__ = f"Dekker double-float (2xFP32) {label} GPU IHS periodogram wrapper (float64 I/O)."

        @staticmethod
        def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
            """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            return _run_ihs_double_prepared(fn_df, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

        @staticmethod
        def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
            """Return automatic double-float IHS frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_ihs_double_prepared(fn_df, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
            return frequency, result

    class ihs:
        __module__ = module_name
        __doc__ = f"Native double-precision (FP64) {label} GPU IHS periodogram wrapper."

        @staticmethod
        def power(device, Nf, df, f0, t, y, dy=None, backend="pswf43", nterms=3, normalization="default"):
            """Return IHS power on ``f0 + df * arange(Nf)`` offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(t, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            return _run_ihs_double_prepared(fn_d, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)

        @staticmethod
        def autopower(device, x, y, dy=None, fmin=None, fmax=12, oversampling=5, normalization="default", nterms=3, backend="pswf43"):
            """Return automatic double-precision IHS frequency and power arrays offloaded to ``device``."""
            device = check_device_id(device)
            normalization = _check_ihs_normalization(normalization)
            backend_id = check_offload_backend_id(backend, label)
            nterms = _check_int("nterms", nterms, 1)
            t_arr, y_arr, dy_arr = _prepare_numpy_inputs(x, y, dy, np.float64)
            _check_observation_count(t_arr.size, nterms)
            delta_t = float(np.max(t_arr) - np.min(t_arr))
            Nf, df, f0 = _auto_grid(delta_t, fmin, fmax, oversampling, nterms=nterms)
            Nf, df, f0, nterms = _check_grid(Nf, df, f0, nterms, min_Nf=8)
            frequency = f0 + df * np.arange(Nf, dtype=np.float64)
            result = _run_ihs_double_prepared(fn_d, device, Nf, df, f0, t_arr, y_arr, dy_arr, backend_id, nterms, normalization)
            return frequency, result

    return ihsh, ihsf, ihsdf, ihs


__all__ = [
    "CDeviceInfo",
    "OFFLOAD_BACKENDS",
    "OffloadDevice",
    "bind_offload_abi",
    "check_device_id",
    "check_offload_backend_id",
    "make_fastchi2_classes",
    "make_ihs_classes",
]
