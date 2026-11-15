"""Release machinery for Toeplitz-LS: per-variant libraries, wheels and checks.

Nothing in this package is shipped in the wheel; it only builds and verifies
it.  Entry points are the ``python -m dispatch.*`` modules:

    dispatch.variants    the target and microarchitecture matrix
    dispatch.toolchains  compiler discovery and one-off provisioning
    dispatch.build       compile one target into .build/<target>/_lib
    dispatch.wheel       assemble and ABI-check the wheel
    dispatch.sdist       stage the source distribution
    dispatch.check       import/run a built wheel (native, wine, qemu)
    dispatch.project     package metadata, read from pyproject.toml
"""
