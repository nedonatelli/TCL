"""Extension-module build configuration.

All package metadata lives in pyproject.toml; this file exists only to
declare the compiled extensions, which setuptools still requires a
setup.py for. The NRLMSISE-00 extension compiles the vendored
public-domain reference C implementation (csrc/nrlmsise00/README.md
has provenance and license notes) into
``pytcl.atmosphere._nrlmsise00_c``; when it cannot be imported at
runtime, ``pytcl.atmosphere.nrlmsise00`` falls back to its validated
pure-Python transcription, so a failed extension build degrades
performance, never correctness.
"""

from setuptools import Extension, setup

# The extension uses only the stable ABI (abi3): one cp311-abi3 wheel
# per platform covers every CPython >= 3.11, current and future --
# no per-minor-version builds, and new CPython releases (3.15+) work
# without a new wheel.
setup(
    ext_modules=[
        Extension(
            name="pytcl.atmosphere._nrlmsise00_c",
            sources=[
                "csrc/nrlmsise00/pymodule.c",
                "csrc/nrlmsise00/nrlmsise-00.c",
                "csrc/nrlmsise00/nrlmsise-00_data.c",
            ],
            include_dirs=["csrc/nrlmsise00"],
            define_macros=[("INLINE", None), ("Py_LIMITED_API", "0x030B0000")],
            py_limited_api=True,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
