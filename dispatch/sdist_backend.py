"""PEP 517 backend used inside the sdist to compile the library at install time.

``dispatch.sdist`` copies this file verbatim into the archive as
``tls_source_build.py``, so it must stay standalone: no imports from the
``dispatch`` package and no third-party dependencies.  It is checked and
formatted with the rest of the tree, but only this file's own module-level code
runs inside the archive.

The wheels are the fast path: they carry one library per microarchitecture
variant and pick one at import.  This backend is the fallback for platforms
without a wheel, and it produces the layout the runtime already understands --
a single ``toeplitz_ls/tls.so``, the "source build" that ``_dispatch.load()``
loads when no manifest is present.

The compile mirrors the ``generic`` target of the repository Makefile, minus
``-march=native``: the result has to run on every machine of the platform, not
just on the one that built it.
"""

from __future__ import annotations

import base64
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = "toeplitz_ls"

# Every translation unit of the library, once per precision (Makefile: LIB_SOURCES).
SCALAR_SOURCES = ("nanofft", "nufft1", "linalg", "chi2per", "utils")
PRECISIONS = (
    ("tlsf", None, ()),
    ("tls", "DOUBLE", ()),
    ("tlsdd", "DOUBLE_DOUBLE", ("-fno-fast-math",)),
)
# Compiled once each: they are precision independent.
SHARED_SOURCES = ("tuning", "cpu_features", "probe_arm64")

CFLAGS = (
    "-std=gnu11",
    "-O3",
    "-ffast-math",
    "-Wall",
    "-Wextra",
    "-fPIC",
    "-fvisibility=hidden",
    "-DTLS_BUILD_SHARED",
    "-DMAX_TWIDDLE_REUSE=16",
)


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def _cc() -> str:
    return os.environ.get("CC") or "cc"


def _run(cmd) -> None:
    result = subprocess.run([str(part) for part in cmd], text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise BuildError("command failed: %s" % " ".join(str(part) for part in cmd))


def compile_library(build_dir: Path) -> Path:
    """Compile and link the baseline library; return the shared object."""
    include = HERE / "include"
    generated = build_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    # chi2per.c includes "scaling.h"; the Makefile stages scaling_generic.h the
    # same way so that a native build can substitute a measured one.
    shutil.copyfile(include / "scaling_generic.h", generated / "scaling.h")

    flags = [*CFLAGS, "-I" + str(include), "-I" + str(generated)]
    objects = []
    for name, define, extra in PRECISIONS:
        for source in SCALAR_SOURCES:
            obj = build_dir / ("%s_%s.o" % (name, source))
            cmd = [_cc(), *flags, *extra]
            if define:
                cmd.append("-D" + define)
            cmd += ["-c", str(HERE / "src" / (source + ".c")), "-o", str(obj)]
            _run(cmd)
            objects.append(obj)
    for source in SHARED_SOURCES:
        obj = build_dir / (source + ".o")
        _run([_cc(), *flags, "-c", str(HERE / "src" / (source + ".c")), "-o", str(obj)])
        objects.append(obj)

    library = build_dir / (PACKAGE + ".so")
    _run([_cc(), *flags, "-shared", "-o", str(library), *objects, "-lm"])
    return library


# ---------------------------------------------------------------------------
# Wheel metadata
# ---------------------------------------------------------------------------
def _pkg_info() -> str:
    return (HERE / "PKG-INFO").read_text(encoding="utf-8")


def _version() -> str:
    for line in _pkg_info().splitlines():
        if line.startswith("Version: "):
            return line[len("Version: ") :].strip()
        if not line.strip():
            break
    raise BuildError("PKG-INFO has no Version field")


def platform_tag() -> str:
    """The wheel tag of the platform this build runs on."""
    system = platform.system()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "arm64" if system == "Darwin" else "aarch64"}.get(machine, machine)

    if system == "Linux":
        return "linux_%s" % machine
    if system == "Darwin":
        release = platform.mac_ver()[0] or "11.0"
        parts = [int(part) for part in release.split(".")[:2]]
        while len(parts) < 2:
            parts.append(0)
        return "macosx_%d_%d_%s" % (parts[0], parts[1], machine)
    raise BuildError(
        "this source distribution only builds on Linux and macOS (%s/%s here); install a prebuilt wheel for your platform instead" % (system, machine)
    )


def _dist_info() -> str:
    return "%s-%s.dist-info" % (PACKAGE, _version())


def _wheel_file() -> str:
    return "%s-%s-py3-none-%s.whl" % (PACKAGE, _version(), platform_tag())


def _record(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return "%s,sha256=%s,%d" % (name, digest, len(data))


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------
def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    dist_info = Path(metadata_directory) / _dist_info()
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_pkg_info(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8")
    return dist_info.name


def _wheel_metadata() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: toeplitz-ls source backend",
            "Root-Is-Purelib: false",
            "Tag: py3-none-%s" % platform_tag(),
            "",
        ]
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    build_dir = HERE / ".tls-build"
    library = compile_library(build_dir)

    dist_info = _dist_info()
    out = Path(wheel_directory) / _wheel_file()
    record: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as wheel:

        def add(name: str, data: bytes) -> None:
            wheel.writestr(name, data)
            record.append(_record(name, data))

        for source in sorted((HERE / PACKAGE).glob("*.py")):
            add("%s/%s" % (PACKAGE, source.name), source.read_bytes())
        add("%s/tls.so" % PACKAGE, library.read_bytes())
        add("%s/METADATA" % dist_info, _pkg_info().encode("utf-8"))
        add("%s/WHEEL" % dist_info, _wheel_metadata().encode("utf-8"))
        add("%s/licenses/LICENSE" % dist_info, (HERE / "LICENSE").read_bytes())
        record.append("%s/RECORD,," % dist_info)
        wheel.writestr("%s/RECORD" % dist_info, "\n".join(record) + "\n")

    return out.name


def build_sdist(sdist_directory, config_settings=None):
    name = "%s-%s.tar.gz" % (PACKAGE, _version())
    out = Path(sdist_directory) / name
    with tarfile.open(out, "w:gz") as archive:
        for item in sorted(HERE.rglob("*")):
            relative = item.relative_to(HERE)
            if ".tls-build" in relative.parts or item.suffix in (".pyc", ".o", ".so", ".whl"):
                continue
            archive.add(item, arcname="%s-%s/%s" % (PACKAGE, _version(), relative), recursive=False)
    return name


if __name__ == "__main__":
    # Usable by hand after unpacking the sdist: `python tls_source_build.py`
    # builds .tls-build/toeplitz_ls.so and reports what it found.
    print("compiling %s for %s" % (PACKAGE, platform_tag()))
    print(compile_library(HERE / ".tls-build"))
