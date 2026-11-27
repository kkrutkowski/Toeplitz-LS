"""Hardware-offloaded periodogram backends for Toeplitz-LS."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
__all__ = ["vulkan"]
if (_HERE / "sycl").is_dir():
    __all__.insert(0, "sycl")
