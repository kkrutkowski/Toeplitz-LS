"""Offload periodogram benchmark adapters (SYCL GPU, CPU native, and future backends)."""

from __future__ import annotations

import math
import operator
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOEPLITZ_PYTHON = ROOT / "toeplitz-ls"
if str(TOEPLITZ_PYTHON) not in sys.path:
    sys.path.insert(0, str(TOEPLITZ_PYTHON))

from toeplitz_ls import ihs as _cpu_ihs  # noqa: E402
from toeplitz_ls import ihsf as _cpu_ihsf  # noqa: E402
from toeplitz_ls import ihsh as _cpu_ihsh  # noqa: E402
from toeplitz_ls import tls as _cpu_tls  # noqa: E402
from toeplitz_ls import tlsf as _cpu_tlsf  # noqa: E402
from toeplitz_ls import tlsh as _cpu_tlsh  # noqa: E402
from toeplitz_ls.offload import sycl as _sycl  # noqa: E402
from toeplitz_ls.offload import vulkan as _vulkan  # noqa: E402

_C_INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class FrequencyGrid:
    """Regular frequency grid matching ``tls.autopower``."""

    frequency: np.ndarray
    Nf: int
    df: float
    f0: float


def trim_mantissa(value: float, bits: int) -> float:
    mantissa, exponent = math.frexp(value)
    mantissa = math.floor(mantissa * (1 << bits)) / (1 << bits)
    return math.ldexp(mantissa, exponent)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    result = operator.index(value)
    if result < 1 or result > _C_INT_MAX:
        raise ValueError(f"{name} must be in [1, {_C_INT_MAX}]")
    return result


def make_frequency_grid(
    t,
    *,
    fmin: float | None = None,
    fmax: float = 4.0,
    oversampling: float = 5.0,
    nterms: int = 1,
) -> FrequencyGrid:
    nterms = _positive_int("nterms", nterms)
    t_arr = np.ascontiguousarray(t, dtype=np.float64)
    delta_t = float(np.max(t_arr) - np.min(t_arr))
    if delta_t <= 0.0:
        raise ValueError("t must span a positive interval")
    df = trim_mantissa(1.0 / (oversampling * delta_t * nterms), 5)
    if fmin is None:
        first_multiple = math.floor((1.0 / delta_t) / df) + 1
    else:
        first_multiple = math.ceil(float(fmin) / df)
    f0 = first_multiple * df
    Nf = _positive_int("Nf", math.floor((float(fmax) - f0) / df) + 1)
    frequency = f0 + df * np.arange(Nf, dtype=np.float64)
    return FrequencyGrid(frequency=frequency, Nf=Nf, df=df, f0=f0)


class OffloadMethod:
    """Base adapter for offload and CPU-reference benchmark methods."""

    name: str = ""
    backend_family: str = "sycl"
    normalization: str = "standard"

    def warmup(self, curve_t, curve_y, curve_dy, nterms: int, *, fmax: float, oversampling: float) -> None:
        grid = make_frequency_grid(curve_t, fmax=fmax, oversampling=oversampling, nterms=nterms)
        self.power(grid.Nf, grid.df, grid.f0, curve_t, curve_y, curve_dy, nterms=nterms)

    def power(self, Nf, df, f0, t, y, dy=None, nterms=1, normalization=None, *, frequency=None):
        raise NotImplementedError


class SyclFastChi2Method(OffloadMethod):
    backend_family = "sycl"

    def __init__(self, name: str, module, *, device: int = 0, pswf_backend: str = "pswf43", normalization: str = "standard"):
        self.name = name
        self.module = module
        self.device = int(device)
        self.pswf_backend = pswf_backend
        self.normalization = normalization

    def power(self, Nf, df, f0, t, y, dy=None, nterms=1, normalization=None, *, frequency=None):
        norm = self.normalization if normalization is None else normalization
        return self.module.power(
            self.device,
            Nf,
            df,
            f0,
            t,
            y,
            dy,
            backend=self.pswf_backend,
            nterms=nterms,
            normalization=norm,
        )


class SyclIhsMethod(OffloadMethod):
    backend_family = "sycl"

    def __init__(self, name: str, module, *, device: int = 0, pswf_backend: str = "pswf43", normalization: str = "default"):
        self.name = name
        self.module = module
        self.device = int(device)
        self.pswf_backend = pswf_backend
        self.normalization = normalization

    def power(self, Nf, df, f0, t, y, dy=None, nterms=1, normalization=None, *, frequency=None):
        norm = self.normalization if normalization is None else normalization
        return self.module.power(
            self.device,
            Nf,
            df,
            f0,
            t,
            y,
            dy,
            backend=self.pswf_backend,
            nterms=nterms,
            normalization=norm,
        )


class CpuFastChi2Method(OffloadMethod):
    backend_family = "cpu"

    def __init__(self, name: str, module, *, pswf_backend: str = "pswf43", normalization: str = "standard"):
        self.name = name
        self.module = module
        self.pswf_backend = pswf_backend
        self.normalization = normalization

    def power(self, Nf, df, f0, t, y, dy=None, nterms=1, normalization=None, *, frequency=None):
        norm = self.normalization if normalization is None else normalization
        kwargs = {"backend": self.pswf_backend, "nterms": nterms, "normalization": norm}
        if nterms > 1:
            kwargs["solver"] = "levinson"
        return self.module.power(Nf, df, f0, t, y, dy, **kwargs)


class CpuIhsMethod(OffloadMethod):
    backend_family = "cpu"

    def __init__(self, name: str, module, *, pswf_backend: str = "pswf43", normalization: str = "default"):
        self.name = name
        self.module = module
        self.pswf_backend = pswf_backend
        self.normalization = normalization

    def power(self, Nf, df, f0, t, y, dy=None, nterms=1, normalization=None, *, frequency=None):
        norm = self.normalization if normalization is None else normalization
        return self.module.power(
            Nf,
            df,
            f0,
            t,
            y,
            dy,
            backend=self.pswf_backend,
            nterms=nterms,
            normalization=norm,
        )


class VulkanFastChi2Method(SyclFastChi2Method):
    backend_family = "vulkan"


class VulkanIhsMethod(SyclIhsMethod):
    backend_family = "vulkan"


def build_registry(device: int = 0, pswf_backend: str = "pswf43", vk_device: int | None = None) -> dict[str, OffloadMethod]:
    """Build the method registry for the selected SYCL and Vulkan device IDs."""
    vk_dev = device if vk_device is None else vk_device
    methods: list[OffloadMethod] = [
        SyclFastChi2Method("sycl_tlsh", _sycl.tlsh, device=device, pswf_backend=pswf_backend),
        SyclFastChi2Method("sycl_tlsf", _sycl.tlsf, device=device, pswf_backend=pswf_backend),
        SyclFastChi2Method("sycl_tlsdf", _sycl.tlsdf, device=device, pswf_backend=pswf_backend),
        SyclFastChi2Method("sycl_tls", _sycl.tls, device=device, pswf_backend=pswf_backend),
        SyclFastChi2Method("sycl_tlsf_regularized", _sycl.tlsf, device=device, pswf_backend=pswf_backend, normalization="regularized"),
        SyclIhsMethod("sycl_ihsh", _sycl.ihsh, device=device, pswf_backend=pswf_backend),
        SyclIhsMethod("sycl_ihsf", _sycl.ihsf, device=device, pswf_backend=pswf_backend),
        SyclIhsMethod("sycl_ihsdf", _sycl.ihsdf, device=device, pswf_backend=pswf_backend),
        SyclIhsMethod("sycl_ihs", _sycl.ihs, device=device, pswf_backend=pswf_backend),
        VulkanFastChi2Method("vk_tlsh", _vulkan.tlsh, device=vk_dev, pswf_backend=pswf_backend),
        VulkanFastChi2Method("vk_tlsf", _vulkan.tlsf, device=vk_dev, pswf_backend=pswf_backend),
        VulkanFastChi2Method("vk_tlsdf", _vulkan.tlsdf, device=vk_dev, pswf_backend=pswf_backend),
        VulkanFastChi2Method("vk_tls", _vulkan.tls, device=vk_dev, pswf_backend=pswf_backend),
        VulkanFastChi2Method("vk_tlsf_regularized", _vulkan.tlsf, device=vk_dev, pswf_backend=pswf_backend, normalization="regularized"),
        VulkanIhsMethod("vk_ihsh", _vulkan.ihsh, device=vk_dev, pswf_backend=pswf_backend),
        VulkanIhsMethod("vk_ihsf", _vulkan.ihsf, device=vk_dev, pswf_backend=pswf_backend),
        VulkanIhsMethod("vk_ihsdf", _vulkan.ihsdf, device=vk_dev, pswf_backend=pswf_backend),
        VulkanIhsMethod("vk_ihs", _vulkan.ihs, device=vk_dev, pswf_backend=pswf_backend),
        CpuFastChi2Method("cpu_tlsh", _cpu_tlsh, pswf_backend=pswf_backend),
        CpuFastChi2Method("cpu_tlsf", _cpu_tlsf, pswf_backend=pswf_backend),
        CpuFastChi2Method("cpu_tls", _cpu_tls, pswf_backend=pswf_backend),
        CpuIhsMethod("cpu_ihsh", _cpu_ihsh, pswf_backend=pswf_backend),
        CpuIhsMethod("cpu_ihsf", _cpu_ihsf, pswf_backend=pswf_backend),
        CpuIhsMethod("cpu_ihs", _cpu_ihs, pswf_backend=pswf_backend),
    ]
    return {m.name: m for m in methods}


__all__ = ["FrequencyGrid", "OffloadMethod", "build_registry", "make_frequency_grid"]
