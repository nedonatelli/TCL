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
            define_macros=[("INLINE", None)],
        )
    ]
)
