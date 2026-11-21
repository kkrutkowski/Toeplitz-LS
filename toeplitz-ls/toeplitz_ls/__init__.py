"""Python interface for the Toeplitz least-squares FastChi2 library."""

from . import _dispatch
from ._fastchi2 import DD, tls, tlsdd, tlsf, trim_mantissa
from ._ihs import ihs, ihsdd, ihsf
from .utils import Peak, Peaks, get_peaks

__version__ = "1.0.0"

#: Manifest entry of the microarchitecture variant this process loaded.
__dispatch__ = _dispatch.selected()

__all__ = [
    "DD",
    "tlsf",
    "tls",
    "tlsdd",
    "ihsf",
    "ihs",
    "ihsdd",
    "trim_mantissa",
    "Peak",
    "Peaks",
    "get_peaks",
    "__version__",
]
