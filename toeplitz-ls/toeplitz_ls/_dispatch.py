"""Import-time selection of the native library variant.

A wheel ships one shared object per microarchitecture level (see
``dispatch/variants.py`` in the source tree) plus ``_lib/manifest.json``
describing them.  This module looks at the CPU it is running on, picks the
widest variant that CPU can execute, and loads it.  The choice is made once,
at import, so a periodogram call pays nothing for the dispatch.

Fallbacks, in order:

* ``TOEPLITZ_LS_DISPATCH=<variant>`` forces a variant (it must be supported by
  the CPU, otherwise the import fails instead of crashing later with SIGILL);
* ``TOEPLITZ_LS_DISPATCH=list`` prints the shipped variants and continues with
  automatic selection;
* a source build (``make native`` / ``make generic``) has no ``_lib`` directory
  and keeps loading the single ``toeplitz_ls/tls.so`` next to this file.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_DIR = HERE / "_lib"
MANIFEST_PATH = LIB_DIR / "manifest.json"
LEGACY_PATH = HERE / "tls.so"

ENV = "TOEPLITZ_LS_DISPATCH"

_LIB = None
_SELECTED = None


# ---------------------------------------------------------------------------
# CPU feature probing
# ---------------------------------------------------------------------------
def _read_proc_cpuinfo_flags():
    flags = set()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                key, sep, value = line.partition(":")
                if sep and key.strip().lower() in ("flags", "features"):
                    flags.update(value.split())
                    break
    except OSError:
        pass
    return flags


def _prctl_sve_vector_length():
    """SVE vector length in bytes via prctl(PR_SVE_GET_VL), or 0."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.prctl(51, 0, 0, 0, 0)  # PR_SVE_GET_VL
    except (OSError, AttributeError):
        return 0
    if rc <= 0:
        return 0
    return rc & 0xFFFF


def _windows_features():
    features = set()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        present = kernel32.IsProcessorFeaturePresent
        present.argtypes = [ctypes.c_uint]
        present.restype = ctypes.c_int
    except (OSError, AttributeError):
        return features
    # Values from the Windows SDK's winnt.h (PF_*_INSTRUCTIONS_AVAILABLE).
    table = {
        36: "ssse3",
        37: "sse4_1",
        38: "sse4_2",
        39: "avx",
        40: "avx2",
        41: "avx512f",
        19: "asimd",
        46: "sve",
    }
    for bit, name in table.items():
        if present(bit):
            features.add(name)
    return features


def _macos_sysctl(name):
    try:
        libc = ctypes.CDLL(None)
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
            return ""
        return buf.value.decode("ascii", "ignore")
    except (OSError, AttributeError):
        return ""


def _macos_features():
    features = set()
    if platform.machine() == "arm64":
        features.add("asimd")  # Apple silicon always has NEON, never SVE
        return features
    leaf = _macos_sysctl("machdep.cpu.features").split()
    leaf7 = _macos_sysctl("machdep.cpu.leaf7_features").split()
    # Both sysctls are space-separated CPUID flag names; match whole tokens
    # ("AVX" must not fire on "AVX2.0", and macOS reports "AVX", not "AVX1.0").
    tokens = set(leaf) | set(leaf7)
    table = {
        "SSE4.1": ("sse4_1",),
        "SSE4.2": ("sse4_2",),
        "SSSE3": ("ssse3",),
        "POPCNT": ("popcnt",),
        "CX16": ("cx16",),
        "LAHF": ("lahf_lm",),
        "AVX": ("avx",),
        "AVX1.0": ("avx",),
        "XSAVE": ("xsave",),
        "F16C": ("f16c",),
        "AVX2": ("avx2",),
        "AVX2.0": ("avx2",),
        "AVX512F": ("avx512f",),
        "AVX512BW": ("avx512bw",),
        "AVX512CD": ("avx512cd",),
        "AVX512DQ": ("avx512dq",),
        "AVX512VL": ("avx512vl",),
        "BMI1": ("bmi1",),
        "BMI2": ("bmi2",),
        "FMA": ("fma",),
        "ABM": ("abm",),
        "LZCNT": ("lzcnt",),
        "MOVBE": ("movbe",),
    }
    for token, names in table.items():
        if token in tokens:
            features.update(names)
    return features


#: Bit order of the mask returned by the library's ``tls_cpu_features()``
#: (include/tls_cpu.h).  Change both together.
_FEATURE_BITS = (
    "sse4_1",
    "sse4_2",
    "ssse3",
    "popcnt",
    "cx16",
    "lahf_lm",
    "avx",
    "xsave",
    "avx2",
    "fma",
    "bmi1",
    "bmi2",
    "f16c",
    "lzcnt",
    "movbe",
    "avx512f",
    "avx512bw",
    "avx512cd",
    "avx512dq",
    "avx512vl",
    "asimd",
    "sve",
    "dotprod",
    "i8mm",
    "bf16",
)


def _probe_features(manifest):
    """Feature names reported by the baseline library's own CPUID/HWCAP probe.

    This is the authoritative answer: it is the same CPU the code will run on,
    and the probe also checks that the OS enabled the AVX/AVX-512 state.  The
    baseline library is always shipped and always safe to load, so asking it
    before choosing a wider variant costs one dlopen.
    """
    variants = manifest.get("variants") or []
    if not variants:
        return set()
    baseline = variants[manifest.get("baseline", 0)]
    try:
        probe_lib = ctypes.CDLL(str(LIB_DIR / baseline["file"]))
        probe = probe_lib.tls_cpu_features
        probe.argtypes = []
        probe.restype = ctypes.c_ulonglong
        mask = int(probe())
    except (OSError, AttributeError):
        return set()
    return {name for bit, name in enumerate(_FEATURE_BITS) if mask >> bit & 1}


def host_features(manifest=None):
    """Set of feature names this CPU can execute (OS hints + the library probe)."""
    features = _probe_features(manifest) if manifest is not None else set()
    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "windows":
        features |= _windows_features()
        if machine in ("arm64", "aarch64"):
            features.add("asimd")  # mandatory on AArch64
        return features

    if system == "darwin":
        features |= _macos_features()
        return features

    flags = set(_read_proc_cpuinfo_flags())
    # Names differ between the Linux flag list and the ISA level names.
    if "abm" in flags:
        flags.add("lzcnt")
    if "sve" in flags:
        vector_bytes = _prctl_sve_vector_length()
        if vector_bytes:
            flags.add("sve_vl:%d" % vector_bytes)
    if machine in ("aarch64", "arm64"):
        flags.add("asimd")  # mandatory on AArch64, even when cpuinfo omits it
    return features | flags


def _complete_sve_length(manifest, features):
    """Ask the baseline library for the SVE vector length when the OS will not.

    Linux answers through prctl and needs no help; Windows publishes only an
    "SVE present" bit, so the length comes from the baseline library's `cntb`
    probe, which is safe to call as soon as that bit is set.
    """
    if "sve" not in features or any(f.startswith("sve_vl:") for f in features):
        return
    variants = manifest.get("variants") or []
    baseline = variants[manifest.get("baseline", 0)] if variants else None
    if baseline is None:
        return
    try:
        probe_lib = ctypes.CDLL(str(LIB_DIR / baseline["file"]))
        probe = probe_lib.tls_probe_sve_vector_length
        probe.argtypes = []
        probe.restype = ctypes.c_ulong
        length = int(probe())
    except (OSError, AttributeError):
        return
    if length > 0:
        features.add("sve_vl:%d" % length)


def _has_any(alternatives, features):
    """True if one of `alternatives` is present; a nested group is itself OR-ed.

    Manifests written by older builds carry level descriptions such as
    ("cx16", "lahf_lm", ...) whose members were meant to be *all* required; a
    nested alternative ("abm", "lzcnt") is the one genuine either-or.  Treating
    a nested sequence as another OR group keeps such manifests readable instead
    of raising "unhashable type" on a set lookup.
    """
    for alt in alternatives:
        if isinstance(alt, (list, tuple)):
            if _has_any(alt, features):
                return True
        elif alt in features:
            return True
    return False


def _satisfied(requires, features):
    return all(_has_any(group, features) for group in requires)


# ---------------------------------------------------------------------------
# Manifest and selection
# ---------------------------------------------------------------------------
def load_manifest():
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError:
        return None


def _select(manifest, features, forced=None):
    variants = manifest.get("variants") or []
    if not variants:
        raise RuntimeError("%s lists no variants" % MANIFEST_PATH)
    if forced:
        for entry in variants:
            if entry["name"] == forced:
                if not _satisfied(entry.get("requires", []), features):
                    raise RuntimeError(
                        "TOEPLITZ_LS_DISPATCH=%s is not supported by this CPU (missing %s); "
                        "shipped variants: %s" % (forced, _missing(entry.get("requires", []), features), ", ".join(v["name"] for v in variants))
                    )
                return entry
        raise RuntimeError("TOEPLITZ_LS_DISPATCH=%s is not shipped; available: %s" % (forced, ", ".join(v["name"] for v in variants)))
    for entry in reversed(variants):
        if _satisfied(entry.get("requires", []), features):
            return entry
    return variants[manifest.get("baseline", 0)]


def _label(group):
    return "|".join(str(item) for item in group)


def _missing(requires, features):
    return ", ".join(_label(group) for group in requires if not _has_any(group, features))


def _describe(manifest, features):
    for entry in manifest.get("variants", []):
        ok = _satisfied(entry.get("requires", []), features)
        print("  %-22s %s%s" % (entry["name"], entry["file"], "" if ok else "  (unsupported on this CPU)"))


def load():
    """Return the shared library for this CPU, loading it once."""
    global _LIB, _SELECTED
    if _LIB is not None:
        return _LIB

    forced = os.environ.get(ENV, "").strip()
    manifest = load_manifest()
    if manifest is None:
        if not LEGACY_PATH.exists():
            raise FileNotFoundError("%s is missing. Build the package from source with 'make native' (or install a wheel) before importing it." % LEGACY_PATH)
        _LIB = ctypes.CDLL(str(LEGACY_PATH))
        _SELECTED = {"name": "source-build", "file": LEGACY_PATH.name, "requires": []}
        return _LIB

    features = host_features(manifest)
    _complete_sve_length(manifest, features)
    if forced.lower() in ("list", "help"):
        print("Toeplitz-LS shipped variants (%s):" % platform.machine())
        _describe(manifest, features)
        forced = ""

    entry = _select(manifest, features, forced or None)
    path = LIB_DIR / entry["file"]
    lib = ctypes.CDLL(str(path))
    _LIB = lib
    _SELECTED = entry
    return lib


def selected():
    """Manifest entry of the loaded variant (or a synthetic source-build one)."""
    if _SELECTED is None:
        load()
    return _SELECTED


def sve_vector_length():
    """Hardware SVE vector length in bytes, probing the loaded library."""
    lib = load()
    try:
        probe = lib.tls_probe_sve_vector_length
    except AttributeError:
        return 0
    probe.argtypes = []
    probe.restype = ctypes.c_ulong
    return int(probe())


__all__ = ["load", "selected", "host_features", "sve_vector_length", "load_manifest", "selected"]
