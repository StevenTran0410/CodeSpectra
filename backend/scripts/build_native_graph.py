"""Build native graph extension in-place (pybind11)."""
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
    src = root / "native" / "graph_native.cpp"
    if not src.exists():
        raise FileNotFoundError(f"Missing native source: {src}")

    is_windows = platform.system().lower().startswith("win")
    compile_args = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]

    ext_modules = [
        Extension(
            "domain.structural_graph._native_graph",
            [str(src)],
            include_dirs=[get_include()],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]

    # Build extension directly into package dir so runtime can import it.
    setup(
        name="codespectra-native-graph",
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
