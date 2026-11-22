# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Build the CUDA-independent pybind11 CPU renderer extension."""

from __future__ import annotations

from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ROOT / "cpu" / "cpu_render_mixed.cpp",
    ROOT / "cpu" / "cpu_render_scalar.cpp",
    ROOT / "cpu" / "cpu_render_avx2.cpp",
    ROOT / "cpu" / "cpu_render_avx512.cpp",
]
BUILD_DIR = ROOT / "build"
OUTPUT = BUILD_DIR / f"cpu_render_native{sysconfig.get_config_var('EXT_SUFFIX')}"


def is_x86() -> bool:
    return platform.machine().lower() in {"x86_64", "amd64", "i386", "i686", "x86"}


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
    compile_flags = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        # Packet implementations use explicit __m256/__m512 intrinsics. Keep
        # auto-vectorization disabled so the scalar reference and post-process
        # loops never silently acquire a higher instruction-set requirement.
        "-fno-vectorize",
        "-fno-slp-vectorize",
        f"-I{pybind11_include}",
        f"-I{sysconfig.get_path('include')}",
    ]
    if sys.platform != "win32":
        # ELF shared objects need position-independent code; without it the
        # link fails on pybind11's thread-local relocations.
        compile_flags.append("-fPIC")
    # AVX2/AVX-512 packet backends exist only for x86. Elsewhere build the
    # module with the scalar backend, which render() then uses.
    portable = {"cpu_render_mixed.cpp", "cpu_render_scalar.cpp"}
    sources = SOURCES if is_x86() else [source for source in SOURCES if source.name in portable]
    objects: list[Path] = []
    for source in sources:
        object_path = BUILD_DIR / f"{source.stem}.obj"
        architecture_flags: list[str] = []
        if not is_x86():
            pass
        elif source.name in {"cpu_render_mixed.cpp", "cpu_render_avx2.cpp"}:
            architecture_flags = ["-mavx2"]
        elif source.name == "cpu_render_avx512.cpp":
            # Keep AVX-512 isolated in this object. Detection and module setup
            # remain executable on AVX2-only systems.
            architecture_flags = ["-mavx512f", "-mavx512dq"]
        command = [
            compiler,
            *compile_flags,
            *architecture_flags,
            "-c",
            str(source),
            "-o",
            str(object_path),
        ]
        print("Compiling", source.name)
        subprocess.run(command, cwd=ROOT, check=True)
        objects.append(object_path)

    # lld is required on Windows. Elsewhere use it when installed and fall
    # back to the system linker otherwise.
    linker_flags = ["-fuse-ld=lld"] if sys.platform == "win32" or shutil.which("ld.lld") else []
    command = [
        compiler,
        "-shared",
        *linker_flags,
        *(str(object_path) for object_path in objects),
        "-o",
        str(OUTPUT),
    ]
    if sys.platform == "darwin":
        # Python symbols resolve from the interpreter at import time.
        command.extend(["-undefined", "dynamic_lookup"])
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
    print("Linking", OUTPUT.name)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
