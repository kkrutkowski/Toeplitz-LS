"""SYCL GPU-offloaded FastChi2/AoVMH(W) least-squares periodogram wrappers.

Mirrors :mod:`toeplitz_ls._fastchi2` with the required SYCL target ``device`` ID
as the first positional argument and no configurable solver/spreading/vectorize
options (degree == 1 uses GLS; degree > 1 uses Hermitian Toeplitz Levinson
recursion on the GPU).

Available precision classes:
- ``tlsh``: IEEE binary16 (FP16) compute with ``float32`` outputs
- ``tlsf``: single-precision (FP32) compute and outputs
- ``tlsdf``: Dekker double-float (2xFP32, ~47-48 bit mantissa) with ``float64`` I/O
- ``tls``: native double-precision (FP64) compute and outputs
"""

from __future__ import annotations

from .._common import check_offload_backend_id, make_fastchi2_classes
from ._lib import load_library, raise_status


def _sycl_backend_id(backend) -> int:
    return check_offload_backend_id(backend, "SYCL")


tlsh, tlsf, tlsdf, tls = make_fastchi2_classes(
    label="SYCL",
    prefix="tls_sycl",
    module_name=__name__,
    load_library=load_library,
    raise_status=raise_status,
)

__all__ = ["tlsh", "tlsf", "tlsdf", "tls"]
