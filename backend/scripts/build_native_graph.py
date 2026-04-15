"""Build native extensions in-place (pybind11).

Builds:
  - domain.structural_graph._native_graph  (graph_native.cpp)
  - domain.retrieval._native_bm25          (bm25_native.cpp)
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

from pybind11 import get_include
try:
    from setuptools import Extension, setup
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing setuptools in current Python environment.\n"
        "Run: uv pip install setuptools\n"
        "If you use venv Python directly, run:\n"
        '  ".venv\\Scripts\\python.exe" -m pip install setuptools'
    ) from exc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    native_dir = root / "native"

    graph_src = native_dir / "graph_native.cpp"
    bm25_src = native_dir / "bm25_native.cpp"

    if not graph_src.exists():
        raise FileNotFoundError(f"Missing native source: {graph_src}")
    if not bm25_src.exists():
        raise FileNotFoundError(f"Missing native source: {bm25_src}")

    is_windows = platform.system().lower().startswith("win")
    compile_args = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]
    pybind_include = get_include()

    ext_modules = [
        Extension(
            "domain.structural_graph._native_graph",
            [str(graph_src)],
            include_dirs=[pybind_include],
            language="c++",
            extra_compile_args=compile_args,
        ),
        Extension(
            "domain.retrieval._native_bm25",
            [str(bm25_src)],
            include_dirs=[pybind_include],
            language="c++",
            extra_compile_args=compile_args,
        ),
    ]

    # Build extensions directly into package dirs so runtime can import them.
    setup(
        name="codespectra-native",
        version="0.0.0",
        ext_modules=ext_modules,
        packages=[],
        py_modules=[],
        include_package_data=False,
        zip_safe=False,
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[native-build] failed: {exc}", file=sys.stderr)
        raise
