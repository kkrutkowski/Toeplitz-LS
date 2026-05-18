"""Python interface for the Toeplitz least-squares FastChi2 library."""

from ._fastchi2 import DD, tls, tlsdd, tlsf, trim_mantissa

__version__ = "0.0.0"

__all__ = ["DD", "tlsf", "tls", "tlsdd", "trim_mantissa", "__version__"]
