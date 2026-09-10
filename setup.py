"""
DDT Build Script
=================
Uses pybind11's build helpers for seamless C++ extension compilation.
Only defines the C++ extension — all metadata lives in pyproject.toml.
"""

import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# C++ source files for the extension module.
cpp_sources = [
    os.path.join("src", "cpp", "ddt_core.cpp"),
    os.path.join("src", "cpp", "pybind_module.cpp"),
]

# Compiler flags for SIMD auto-vectorization.
extra_compile_args = []
if os.name == "nt":  # Windows / MSVC
    extra_compile_args = ["/O2", "/std:c++17"]
else:  # Linux / macOS
    extra_compile_args = ["-O3", "-march=native", "-std=c++17"]

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
