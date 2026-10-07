"""
DDT Build Script.
=================
Uses pybind11's build helpers for seamless C++ extension compilation.
Only defines the C++ extension — all metadata lives in pyproject.toml.
"""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# C++ source files for the extension module.
cpp_sources = [
    os.path.join("src", "cpp", "ddt_core.cpp"),
    os.path.join("src", "cpp", "pybind_module.cpp"),
]

# Compiler flags.
#   - The code relies on compiler auto-vectorization only; there are no SIMD
#     intrinsics and no OpenMP (the GIL is released in the bindings instead).
#   - Default builds are portable. Set DDT_NATIVE=1 to opt in to host-specific
#     code generation (-march=native / /arch:AVX2). Wheels built that way are
#     NOT portable to other CPUs.
extra_compile_args = []
_native = os.environ.get("DDT_NATIVE", "0") == "1"
if os.name == "nt":  # Windows / MSVC
    extra_compile_args = ["/O2", "/std:c++17"]
    if _native:
        extra_compile_args.append("/arch:AVX2")
else:  # Linux / macOS
    extra_compile_args = ["-O3", "-std=c++17"]
    if _native:
        extra_compile_args.append("-march=native")

ext_modules = [
    Pybind11Extension(
        "ddt._ddt_core",
        sources=cpp_sources,
        include_dirs=[os.path.join("src", "cpp")],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

setup(
    packages=["ddt"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
