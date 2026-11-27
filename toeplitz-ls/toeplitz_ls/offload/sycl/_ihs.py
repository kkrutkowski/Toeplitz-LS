"""SYCL GPU-offloaded incoherent harmonic sum (IHS) periodogram wrappers.

Mirrors :mod:`toeplitz_ls._ihs` with the required SYCL target ``device`` ID
as the first positional argument.
"""

from __future__ import annotations

from .._common import make_ihs_classes
from ._lib import load_library, raise_status

ihsh, ihsf, ihsdf, ihs = make_ihs_classes(
    label="SYCL",
    prefix="tls_sycl",
    module_name=__name__,
    load_library=load_library,
    raise_status=raise_status,
)

__all__ = ["ihsh", "ihsf", "ihsdf", "ihs"]
