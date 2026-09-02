# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Compare the scalar and forced full-AVX2 ray paths with one fixed workload."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import render_cpu
from cfg import offline as config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ssaa", type=int, default=8)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if min(args.width, args.height, args.ssaa, args.workers) <= 0:
        parser.error("width, height, ssaa, and workers must be positive")

    if not render_cpu.native_extension_exists():
        render_cpu.build_native_extension()
    native = render_cpu._native_module()
    if not native.avx2_supported():
        raise RuntimeError("AVX2 is not available on this machine")

    print(f"Python: {sys.executable}", flush=True)
    print(f"Extension: {native.__file__}", flush=True)
    print(f"Native module: {native.__doc__}", flush=True)
    print("Loading configured textures...", flush=True)
    background = render_cpu._load_background_rgba(ROOT / config.skybox_path)
    prebaked_disk = render_cpu._as_gpu_half_texture(
        np.load(ROOT / config.prebaked_disk_path, mmap_mode="r")
    )
    color_lut = render_cpu._as_gpu_half_texture(np.load(ROOT / config.color_lut_path))
    forward, right, up = render_cpu._camera_vectors(config.cam_yaw, config.cam_pitch, config.cam_roll)

    flags = render_cpu.RenderFlag(0)
    if getattr(config, "USE_RK4", False):
        flags |= render_cpu.RenderFlag.USE_RK4
    if not getattr(config, "NO_DEPTH_JITTER", False):
        flags |= render_cpu.RenderFlag.DEPTH_JITTER
    if getattr(config, "RAND_SAMP_DISK", False):
        flags |= render_cpu.RenderFlag.RAND_SAMP_DISK
    if not getattr(config, "NO_BKGD_DOPPLER", False):
        flags |= render_cpu.RenderFlag.BACKGROUND_DOPPLER

    python_params = render_cpu.RenderParams(
        time=float(config.start_t),
        camera_position=config.cam_pos_init,
        forward=forward,
        right=right,
        up=up,
        velocity=(float(config.vx), float(config.vy), float(config.vz)),
        physwidth=args.width * float(config.PIXEL_PITCH),
        physheight=args.height * float(config.PIXEL_PITCH),
        focal_length=float(config.focal_length),
        step=float(config.step),
        maxstep=int(config.maxstep),
        jitternum=args.ssaa,
        frames=1,
        flags=flags,
    )
    params = python_params._native()
    tile_width, tile_height = render_cpu.CPU_TEST_TILE_SIZE
    common_args = (
        args.width,
        args.height,
        background,
        prebaked_disk,
        color_lut,
        params,
        tile_width,
        tile_height,
        args.workers,
    )
    print(
        f"A/B workload: {args.width}x{args.height}, SSAA {args.ssaa}, "
        f"maxstep {params.maxstep}, workers {args.workers}, tile {tile_width}x{tile_height}",
        flush=True,
    )

    print("A: scalar reference started", flush=True)
    started = time.perf_counter()
    scalar = native._render_new_scalar_reference(*common_args)
    scalar_seconds = time.perf_counter() - started
    print(f"A: scalar reference finished in {scalar_seconds:.3f}s", flush=True)

    print("B: forced full-AVX2 path started", flush=True)
    started = time.perf_counter()
    avx2 = native._render_new_avx2_full_math(*common_args)
    avx2_seconds = time.perf_counter() - started
    print(f"B: forced full-AVX2 path finished in {avx2_seconds:.3f}s", flush=True)

    pixel_error = np.max(np.abs(scalar - avx2), axis=2)
    print(f"Speedup: {scalar_seconds / avx2_seconds:.3f}x", flush=True)
    print(
        "RAW error: "
        f"median={np.median(pixel_error):.8g}, "
        f"p99={np.quantile(pixel_error, 0.99):.8g}, "
        f"p99.9={np.quantile(pixel_error, 0.999):.8g}, "
        f"max={np.max(pixel_error):.8g}, "
        f">1e-3={np.count_nonzero(pixel_error > 1e-3)}/{pixel_error.size}",
        flush=True,
    )

    print("C: public auto-dispatched RGBA8 pipeline started", flush=True)
    started = time.perf_counter()
    render_cpu.render_rgba8_new(
        args.width,
        args.height,
        background,
        prebaked_disk,
        color_lut,
        python_params,
        tile_size=(tile_width, tile_height),
        workers=args.workers,
        postprocess=render_cpu.PostprocessParams(
            bloom_threshold=12.0,
            use_aces=bool(getattr(config, "USE_ACES", False)),
            not_use_s_curve=bool(getattr(config, "NOT_USE_S_CURVE", False)),
        ),
    )
    pipeline_seconds = time.perf_counter() - started
    print(
        f"C: full RGBA8 pipeline finished in {pipeline_seconds:.3f}s "
        f"(approximately {pipeline_seconds - avx2_seconds:.3f}s beyond the RAW AVX2 pass)",
        flush=True,
    )


if __name__ == "__main__":
    main()
