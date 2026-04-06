"""Native toolchain detection helpers."""
from __future__ import annotations

import importlib
import shutil


def detect_cpp_toolchain() -> str | None:
    """
    Detect native C++ acceleration status.

    Checks in order:
    1. Native .pyd module loads successfully → most reliable signal
    2. cl.exe / g++ / clang++ in PATH → compiler available but may not have built yet
    """
    try:
        mod = importlib.import_module("domain.structural_graph._native_graph")
        # Verify the key functions are present
        funcs = [f for f in ("compute_scores", "compute_scc", "scan_keywords_bulk") if hasattr(mod, f)]
        return f"native ({len(funcs)}/{3} fns)"
    except Exception:
        pass

    # Compiler fallback — useful on CI/build machines where module isn't deployed yet
    if shutil.which("cl"):
        return "msvc (module not built)"
    if shutil.which("g++"):
        return "g++ (module not built)"
    if shutil.which("clang++"):
        return "clang++ (module not built)"
    return None
