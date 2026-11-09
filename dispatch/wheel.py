"""Assemble a wheel from one built target and check its ABI tags.

The package is pure ctypes, so a single ``py3-none-<platform>`` wheel covers
every Python version: the shared objects are loaded with ``ctypes.CDLL`` and
nothing depends on the CPython ABI.

The platform tag is earned, not declared: :func:`check_abi` re-derives it from
the built binaries (glibc symbol versions, musl NEEDED entries, PE machine and
imports, Mach-O build version and signature) and the wheel is only written when
the binary actually satisfies the tag.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from . import project as P
from . import variants as V

ROOT = V.repo_root()
BUILD = ROOT / ".build"
DIST = ROOT / "dist"
PACKAGE = ROOT / "toeplitz-ls" / "toeplitz_ls"

PACKAGE_FILES = ("__init__.py", "_fastchi2.py", "_ihs.py", "_dispatch.py", "utils.py")
LICENSE = P.LICENSE

# glibc symbol versions the wheels are allowed to need.  Everything the library
# calls (memcpy, malloc, exp, lgamma, clock_gettime via the tools only, ...) is
# at or below this, which is what manylinux_2_17 promises.
GLIBC_MAX = (2, 17)


class AbiError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tool(*names):
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(cmd):
    return subprocess.run([str(c) for c in cmd], text=True, capture_output=True)


def metadata(version: str) -> str:
    return P.metadata(version)


def read_version() -> str:
    return P.version()


# ---------------------------------------------------------------------------
# ABI checks
# ---------------------------------------------------------------------------
def check_abi(library: Path, target) -> list[str]:
    """Return human-readable findings; raise AbiError when the tag is not met."""
    notes = []
    if target.os == "linux" and target.libc == "glibc":
        readelf = _tool("llvm-readelf", "readelf")
        if not readelf:
            raise AbiError("readelf is required to verify the manylinux tag")
        dynamic = _run([readelf, "-d", library]).stdout
        needed = re.findall(r"Shared library: \[([^\]]+)\]", dynamic)
        allowed = {"libc.so.6", "libm.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1"}
        unexpected = [n for n in needed if n not in allowed]
        if unexpected:
            raise AbiError("%s needs non-system libraries: %s" % (library.name, ", ".join(unexpected)))
        symbols = _run([readelf, "--dyn-syms", "-W", library]).stdout
        vector_math = [m for m in re.findall(r"UND\s+(\S+)", symbols) if m.startswith("_ZGV")]
        if vector_math:
            raise AbiError("%s references libmvec (%s), which glibc 2.17 does not have" % (library.name, ", ".join(sorted(set(vector_math))[:3])))
        versions = sorted({tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_(\d+\.\d+)", symbols)})
        if versions and versions[-1] > GLIBC_MAX:
            raise AbiError("%s requires GLIBC_%s but the tag promises %s" % (library.name, versions[-1], ".".join(map(str, GLIBC_MAX))))
        notes.append("glibc <= %s, NEEDED %s" % (".".join(map(str, versions[-1])) if versions else "-", ",".join(needed)))

    elif target.os == "linux" and target.libc == "musl":
        readelf = _tool("llvm-readelf", "readelf")
        dynamic = _run([readelf, "-d", library]).stdout
        needed = re.findall(r"Shared library: \[([^\]]+)\]", dynamic)
        expected = "libc.musl-%s.so.1" % target.arch
        if needed != [expected]:
            raise AbiError("%s must need exactly %s, has %s" % (library.name, expected, needed))
        symbols = _run([readelf, "--dyn-syms", "-W", library]).stdout
        vector_math = [m for m in re.findall(r"UND\s+(\S+)", symbols) if m.startswith("_ZGV")]
        if vector_math:
            raise AbiError("%s references libmvec (%s), which musl does not have" % (library.name, ", ".join(sorted(set(vector_math))[:3])))
        notes.append("musl: NEEDED %s" % expected)

    elif target.os == "windows":
        readobj = _tool("llvm-readobj")
        if not readobj:
            raise AbiError("llvm-readobj is required to verify the Windows DLL")
        headers = _run([readobj, "--file-headers", library]).stdout
        machine = "ARM64" if target.arch == "aarch64" else "AMD64"
        if machine not in headers:
            raise AbiError("%s is not a %s PE" % (library.name, machine))
        imports = _run([readobj, "--coff-imports", library]).stdout
        names = set(re.findall(r"Name: (\S+)", imports))
        allowed_prefix = ("KERNEL32.dll", "api-ms-win-core-", "api-ms-win-crt-", "ucrtbase.dll")
        unexpected = [n for n in names if not n.startswith(allowed_prefix)]
        if unexpected:
            raise AbiError("%s imports a non-system DLL: %s" % (library.name, ", ".join(sorted(unexpected))))
        exports = _run([readobj, "--coff-exports", library]).stdout
        exported = re.findall(r"Name: (\S+)", exports)
        if len(exported) < 50:
            raise AbiError("%s exports only %d symbols" % (library.name, len(exported)))
        if any(name.startswith("_") or "@" in name for name in exported):
            raise AbiError("%s exports decorated names" % library.name)
        notes.append("%s PE, %d undecorated exports, imports %s" % (machine, len(exported), ",".join(sorted(names))))

    elif target.os == "macos":
        objdump = _tool("llvm-objdump")
        if not objdump:
            raise AbiError("llvm-objdump is required to verify the Mach-O")
        headers = _run([objdump, "--macho", "--private-headers", library]).stdout
        minos = re.search(r"minos (\d+\.\d+)", headers)
        if minos and tuple(int(x) for x in minos.group(1).split(".")) > (11, 0):
            # The tag promises 11.0, so a binary that requires 14.0 would not
            # run on the systems the wheel claims to support.
            raise AbiError("%s requires macOS %s but the tag promises 11.0" % (library.name, minos.group(1)))
        if target.arch == "aarch64" and "LC_CODE_SIGNATURE" not in headers:
            raise AbiError("%s has no code signature (required to load on arm64 macOS)" % library.name)
        dylibs = set(re.findall(r"name (/usr/lib/[^ ]+)", headers))
        # /usr/lib/libSystem carries libc/libm; /usr/lib/dyld is the loader
        # itself and always present.
        unexpected = [d for d in dylibs if not (d.startswith("/usr/lib/libSystem") or d == "/usr/lib/dyld")]
        if unexpected:
            raise AbiError("%s links non-system dylibs: %s" % (library.name, unexpected))
        notes.append("Mach-O %s, minos %s%s" % (target.arch, minos.group(1) if minos else "?", ", signed" if "LC_CODE_SIGNATURE" in headers else ""))
    else:
        raise AbiError("no ABI check for target %s" % target.name)
    return notes


def check_staging(staging: Path, target) -> list[str]:
    manifest = json.loads((staging / "manifest.json").read_text())
    notes = []
    names = [entry["name"] for entry in manifest["variants"]]
    expected = [v.name for v in target.variants]
    if names != expected:
        raise AbiError("manifest variants %s != expected %s" % (names, expected))
    baseline = manifest["variants"][manifest.get("baseline", 0)]
    if baseline.get("requires"):
        raise AbiError("the baseline variant must not require anything")
    for entry in manifest["variants"]:
        path = staging / entry["file"]
        if not path.exists():
            raise AbiError("manifest lists a missing library: %s" % entry["file"])
        notes.extend("%s: %s" % (entry["name"], note) for note in check_abi(path, target))
    return notes


# ---------------------------------------------------------------------------
# Wheel assembly
# ---------------------------------------------------------------------------
def _record_line(archive: Path) -> str:
    digest = hashlib.sha256(archive.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return "sha256=%s,%d" % (encoded, archive.stat().st_size)


def build_wheel(target_name: str, version: str | None = None) -> Path:
    target = V.TARGETS[target_name]
    staging = BUILD / target_name / "_lib"
    if not staging.exists():
        raise AbiError("%s has not been built yet (run dispatch.build first)" % target_name)

    notes = check_staging(staging, target)
    for note in notes:
        print("  [abi] " + note)

    version = version or read_version()
    dist_info = "toeplitz_ls-%s.dist-info" % version
    wheel_name = "toeplitz_ls-%s-py3-none-%s.whl" % (version, target.platform_tag)
    DIST.mkdir(exist_ok=True)
    out = DIST / wheel_name
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        record: list[str] = []

        def add(name: str, data: bytes) -> None:
            wheel.writestr(name, data)
            record.append("%s,%s" % (name, _record_line_from_bytes(data)))

        for name in PACKAGE_FILES:
            source = PACKAGE / name
            if not source.exists():
                raise AbiError("package file missing: %s" % source)
            add("toeplitz_ls/" + name, source.read_bytes())
        for item in sorted(staging.iterdir()):
            if item.is_file():
                add("toeplitz_ls/_lib/" + item.name, item.read_bytes())

        add(
            "%s/METADATA" % dist_info,
            metadata(version).encode("utf-8"),
        )
        add(
            "%s/WHEEL" % dist_info,
            "\n".join(
                [
                    "Wheel-Version: 1.0",
                    "Generator: toeplitz-ls dispatch builder",
                    "Root-Is-Purelib: false",
                    "Tag: py3-none-%s" % target.platform_tag,
                    "",
                ]
            ).encode("utf-8"),
        )
        add("%s/licenses/LICENSE" % dist_info, LICENSE.read_bytes())
        record.append("%s/RECORD,," % dist_info)
        wheel.writestr("%s/RECORD" % dist_info, "\n".join(record) + "\n")

    print("[wheel] %s (%.1f KiB)" % (out, out.stat().st_size / 1024.0))
    return out


def _record_line_from_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return "sha256=%s,%d" % (encoded, len(data))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Assemble wheels from built targets.")
    parser.add_argument("--target", action="append", dest="targets", choices=sorted(V.TARGETS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true", help="only run the ABI checks")
    args = parser.parse_args(argv)

    names = args.targets or []
    if args.all and not names:
        names = sorted(V.TARGETS)
    if not names:
        parser.error("give --target, --all or --check")

    version = read_version()
    for name in names:
        if args.check:
            staging = BUILD / name / "_lib"
            if not staging.exists():
                print("[%s] not built" % name)
                continue
            for note in check_staging(staging, V.TARGETS[name]):
                print("[%s] %s" % (name, note))
        else:
            build_wheel(name, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
