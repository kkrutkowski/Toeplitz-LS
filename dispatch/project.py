"""Package metadata, read once from ``pyproject.toml``.

The distributions are assembled by hand: the C libraries come from
``dispatch.build`` and the archives from ``dispatch.wheel`` / ``dispatch.sdist``,
not from a setuptools extension.  This module is what keeps their ``METADATA``
honest -- name, version, summary, license, dependencies, classifiers and the
long description are read from ``toeplitz-ls/pyproject.toml`` and the root
``README.md``, so a wheel cannot advertise one license while the archive next to
it ships another.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "toeplitz-ls" / "pyproject.toml"
README = ROOT / "README.md"
LICENSE = ROOT / "LICENSE"


class MetadataError(RuntimeError):
    pass


def _tomllib():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            raise MetadataError("reading %s needs Python 3.11+ or the 'tomli' package" % PYPROJECT) from None
    return tomllib


def load() -> dict:
    with open(PYPROJECT, "rb") as handle:
        return _tomllib().load(handle)


def project() -> dict:
    try:
        return load()["project"]
    except KeyError:
        raise MetadataError("%s has no [project] table" % PYPROJECT) from None


def version() -> str:
    try:
        return str(project()["version"])
    except KeyError:
        raise MetadataError("%s has no project.version" % PYPROJECT) from None


def license_id() -> str:
    """The license text, from either the SPDX string or the legacy table form."""
    value = project().get("license")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    raise MetadataError('%s must declare a license, e.g. license = { text = "MIT" }; got %r' % (PYPROJECT, value))


def dist_info(version_: str | None = None) -> str:
    return "%s-%s.dist-info" % (str(project()["name"]).replace("-", "_"), version_ or version())


def metadata(version_: str | None = None) -> str:
    """The wheel/sdist ``METADATA``, header block plus the README body."""
    proj = project()
    lines = [
        "Metadata-Version: 2.1",
        "Name: %s" % proj["name"],
        "Version: %s" % (version_ or version()),
    ]
    if proj.get("description"):
        lines.append("Summary: %s" % " ".join(str(proj["description"]).split()))
    lines.append("License: %s" % license_id())

    for author in proj.get("authors", []):
        if author.get("name"):
            lines.append("Author: %s" % author["name"])
        if author.get("email"):
            lines.append("Author-email: %s" % author["email"])

    if proj.get("keywords"):
        lines.append("Keywords: %s" % ",".join(proj["keywords"]))
    if proj.get("requires-python"):
        lines.append("Requires-Python: %s" % proj["requires-python"])
    for classifier in proj.get("classifiers", []):
        lines.append("Classifier: %s" % classifier)
    for label, url in (proj.get("urls") or {}).items():
        lines.append("Project-URL: %s, %s" % (label, url))
    for dependency in proj.get("dependencies", []):
        lines.append("Requires-Dist: %s" % dependency)
    for extra, dependencies in (proj.get("optional-dependencies") or {}).items():
        lines.append("Provides-Extra: %s" % extra)
        for dependency in dependencies:
            lines.append('Requires-Dist: %s; extra == "%s"' % (dependency, extra))

    lines.append("Description-Content-Type: text/markdown")
    lines.append("")
    return "\n".join(lines) + "\n" + README.read_text(encoding="utf-8")
