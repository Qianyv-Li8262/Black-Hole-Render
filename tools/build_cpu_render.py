"""Build the CUDA-independent pybind11 CPU renderer extension."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "cpu" / "cpu_render.cpp"
BUILD_DIR = ROOT / "build"
OUTPUT = BUILD_DIR / f"cpu_render_native{sysconfig.get_config_var('EXT_SUFFIX')}"


def main() -> None:
    compiler = os.environ.get("CXX") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("clang++ was not found. Install LLVM or set CXX to a C++17 compiler.")

    include_override = os.environ.get("PYBIND11_INCLUDE_DIR")
    if include_override:
        pybind11_include = Path(include_override)
    else:
        try:
            import pybind11
        except ModuleNotFoundError as error:
            raise SystemExit("pybind11 was not found. Install it with `pip install pybind11`.") from error
        pybind11_include = Path(pybind11.get_include())
    if not (pybind11_include / "pybind11" / "pybind11.h").is_file():
        raise SystemExit(
            "pybind11 headers were not found. Set PYBIND11_INCLUDE_DIR to the directory containing pybind11/."
        )

    BUILD_DIR.mkdir(exist_ok=True)
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-shared",
        "-fuse-ld=lld",
        f"-I{pybind11_include}",
        f"-I{sysconfig.get_path('include')}",
        str(SOURCE),
        "-o",
        str(OUTPUT),
    ]
    if sys.platform == "win32":
        library_dir = Path(sysconfig.get_config_var("LIBDIR") or Path(sys.prefix) / "libs")
        python_library = library_dir / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        if not python_library.exists():
            raise SystemExit(f"Python import library was not found: {python_library}")
        command.extend([
            str(python_library),
            "-Xlinker",
            f"/IMPLIB:{BUILD_DIR / 'cpu_render_native.lib'}",
        ])
    print("Building", OUTPUT.name)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
