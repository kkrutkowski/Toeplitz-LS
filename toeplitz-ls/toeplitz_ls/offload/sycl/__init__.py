"""SYCL GPU-offloaded periodogram interface for Toeplitz-LS."""

from ._fastchi2 import tls, tlsdf, tlsf, tlsh
from ._ihs import ihs, ihsdf, ihsf, ihsh
from ._lib import SyclDevice, list_devices

__all__ = [
    "SyclDevice",
    "list_devices",
    "tlsh",
    "tlsf",
    "tlsdf",
    "tls",
    "ihsh",
    "ihsf",
    "ihsdf",
    "ihs",
]
