"""Native toolchain detection helpers."""
from __future__ import annotations

import shutil


def detect_cpp_toolchain() -> str | None:
    """
    Detect an available C/C++ compiler on PATH.

    Priority on Windows:
    - MSVC cl.exe
    - MinGW g++
    - clang++
    """
    if shutil.which("cl"):
        return "msvc"
    if shutil.which("g++"):
        return "mingw-g++"
    if shutil.which("clang++"):
        return "clang++"
    return None
