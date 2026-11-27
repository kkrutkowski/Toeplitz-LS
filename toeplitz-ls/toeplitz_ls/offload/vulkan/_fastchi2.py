"""Vulkan/GLSL compute-offloaded FastChi2/AoVMH(W) least-squares periodogram wrappers.

Mirrors :mod:`toeplitz_ls.offload.sycl` with the required Vulkan target ``device`` ID
as the first positional argument.

Available precision classes:
- ``tlsh``: IEEE binary16 (FP16) compute with ``float32`` outputs (requires ``shaderFloat16``;
  if unsupported, raises ``ValueError`` recommending FP32 ``tlsf``)
- ``tlsf``: single-precision (FP32) compute and outputs
- ``tlsdf``: Dekker double-float (2xFP32, ~47-48 bit mantissa) with ``float64`` I/O
- ``tls``: native double-precision (FP64) compute and outputs (requires ``shaderFloat64``;
  if unsupported, raises ``ValueError`` recommending Dekker double-float ``tlsdf``)
"""

from __future__ import annotations

from .._common import make_fastchi2_classes
from ._lib import load_library, raise_status

tlsh, tlsf, tlsdf, tls = make_fastchi2_classes(
    label="Vulkan",
    prefix="tls_vk",
    module_name=__name__,
    load_library=load_library,
    raise_status=raise_status,
)

__all__ = ["tlsh", "tlsf", "tlsdf", "tls"]
