"""ctypes loader and device discovery for the SYCL offload shared library (tls_sycl.so)."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from .._common import CDeviceInfo, OffloadDevice, bind_offload_abi

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
ENV_LIB = "TOEPLITZ_LS_SYCL_LIB"

_STATUS_MESSAGES = {
    -1: "invalid argument",
    -2: "invalid backend",
    -3: "allocation failure",
    -4: "degenerate input",
    -5: "invalid solver",
    -6: "invalid or unsupported SYCL device",
    -7: "SYCL runtime error",
}


@dataclass(frozen=True)
class SyclDevice(OffloadDevice):
    """Metadata describing an available SYCL offload target."""


_LIB_LOCK = threading.Lock()
_LIB = None


def _candidate_library_paths() -> list[Path]:
    paths: list[Path] = []
    env_path = os.environ.get(ENV_LIB, "").strip()
    if env_path:
        paths.append(Path(env_path))
    paths.append(HERE / "tls_sycl.so")
    paths.append(HERE / "_lib" / "tls_sycl.so")
    paths.append(REPO_ROOT / "offload" / "sycl" / "build" / "tls_sycl.so")
    return paths


def _resolve_library_path() -> Path:
    candidates = _candidate_library_paths()
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"tls_sycl.so not found (searched: {searched}). Build it with 'make -C offload/sycl -j$(nproc)' or set TOEPLITZ_LS_SYCL_LIB.")


def load_library():
    """Load tls_sycl.so once (thread-safe) and declare all ctypes signatures."""
    global _LIB
    if _LIB is not None:
        return _LIB

    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB

        lib_path = _resolve_library_path()
        lib = ctypes.CDLL(str(lib_path))
        bind_offload_abi(lib, "tls_sycl")
        _LIB = lib
        return lib


def raise_status(name: str, status: int) -> None:
    """Translate negative C ABI status codes into Python exceptions."""
    if status >= 0:
        return
    lib = load_library()
    raw_err = lib.tls_sycl_last_error()
    detail = raw_err.decode("utf-8", errors="replace").strip() if raw_err else ""
    base = _STATUS_MESSAGES.get(status, f"status {status}")
    msg = f"{name} failed: {base}" + (f" ({detail})" if detail else "")
    if status == -1 or status == -2 or status == -6:
        raise ValueError(msg)
    raise RuntimeError(msg)


def list_devices(*, print_table: bool = True) -> list[SyclDevice]:
    """List available SYCL offload target devices (GPUs sorted first, starting at ID 0).

    Parameters
    ----------
    print_table : bool, default True
        When True, prints a formatted summary table of device IDs and names.

    Returns
    -------
    list[SyclDevice]
        Ordered list of available SYCL devices with their integer IDs.
    """
    lib = load_library()
    count = int(lib.tls_sycl_device_count())
    devices: list[SyclDevice] = []
    info = CDeviceInfo()
    for dev_id in range(count):
        rc = lib.tls_sycl_device_info(dev_id, ctypes.byref(info))
        if rc != 0:
            continue
        devices.append(
            SyclDevice(
                id=int(info.id),
                name=info.name.decode("utf-8", errors="replace"),
                backend=info.backend.decode("utf-8", errors="replace"),
                device_type=info.device_type.decode("utf-8", errors="replace"),
                max_compute_units=int(info.max_compute_units),
                global_mem_size=int(info.global_mem_size),
                is_gpu=bool(info.is_gpu),
                has_fp64=bool(info.has_fp64),
                has_fp16=bool(info.has_fp16),
            )
        )
    if print_table:
        if not devices:
            print("No SYCL devices found.")
        else:
            for d in devices:
                print(d)
    return devices


__all__ = ["SyclDevice", "list_devices", "load_library", "raise_status"]
