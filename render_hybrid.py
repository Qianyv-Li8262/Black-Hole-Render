# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""GPU/CPU frame-parallel entry point for the fixed-camera renderer.

Each GPU owns one renderer instance and claims the lowest-numbered pending
frame under a lock. Frames are independent: camera pose and velocity stay
fixed while scene time advances linearly by ``d_tau`` per frame.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import cv2
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    # Keep --help and argument validation usable in environments where the
    # GPU runtime is not installed. main() still requires CuPy to render.
    cp = None

from cfg import offline as cfg
from helpers.render_manifest import finish_manifest, start_manifest

if cp is not None:
    from helpers.cuda_tex import (
        create_texture_array_2d,
        create_texture_array_3d,
        create_texture_surface_union_2d,
    )


PENDING = np.int8(0)
RUNNING = np.int8(1)
DONE = np.int8(2)
FAILED = np.int8(-1)
BASE_PATH = Path(__file__).resolve().parent


class FrameScheduler:
    """Thread-safe, lowest-frame-first scheduler shared by GPU workers."""

    def __init__(self, total_frames: int, output_dir: Path, overwrite: bool):
        self.status = np.full(total_frames, PENDING, dtype=np.int8)
        self.lock = threading.Lock()
        self.errors: list[tuple[str, int, str]] = []

        if not overwrite:
            for frame_idx in range(1, total_frames + 1):
                path = output_dir / f"frame_{frame_idx:05d}.png"
                if path.is_file() and path.stat().st_size > 0:
                    self.status[frame_idx - 1] = DONE

    def claim(self) -> int | None:
        with self.lock:
            pending = np.flatnonzero(self.status == PENDING)
            if pending.size == 0:
                return None
            frame_idx = int(pending[0])
            self.status[frame_idx] = RUNNING
            return frame_idx + 1

    def finish(self, frame_idx: int) -> None:
        with self.lock:
            self.status[frame_idx - 1] = DONE

    def fail(self, worker_name: str, frame_idx: int, error: str) -> None:
        with self.lock:
            self.status[frame_idx - 1] = FAILED
            self.errors.append((worker_name, frame_idx, error))

    def record_worker_error(self, worker_name: str, error: str) -> None:
        with self.lock:
            self.errors.append((worker_name, 0, error))

    def summary(self) -> dict[str, int]:
        with self.lock:
            return {
                "pending": int(np.count_nonzero(self.status == PENDING)),
                "running": int(np.count_nonzero(self.status == RUNNING)),
                "done": int(np.count_nonzero(self.status == DONE)),
                "failed": int(np.count_nonzero(self.status == FAILED)),
            }


def read_image_bgr(path: Path) -> np.ndarray | None:
    """Load EXR through OpenEXR while retaining OpenCV BGR layout."""
    if path.suffix.lower() != ".exr":
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as exr:
        channels = exr.channels()
        missing = [name for name in ("R", "G", "B") if name not in channels]
        if missing:
            raise RuntimeError(
                f"EXR is missing RGB channels {missing}; available: {list(channels)}"
            )
        return np.ascontiguousarray(
            np.stack(
                (channels["B"].pixels, channels["G"].pixels, channels["R"].pixels),
                axis=-1,
            ),
            dtype=np.float32,
        )


class GpuRenderer:
    """All CUDA resources owned by one GPU worker thread."""

    def __init__(self, gpu_id: int, time_step: float, output_dir: Path):
        self.gpu_id = gpu_id
        self.time_step = np.float32(time_step)
        self.output_dir = output_dir

        with cp.cuda.Device(gpu_id):
            self._initialize()

    def _initialize(self) -> None:
        self.w, self.h = cfg.w, cfg.h
        self.block_x, self.block_y = cfg.block_x, cfg.block_y
        self.grid_x = (self.w + self.block_x - 1) // self.block_x
        self.grid_y = (self.h + self.block_y - 1) // self.block_y

        print(f"[GPU {self.gpu_id}] Loading skybox...")
        img_bgr = read_image_bgr(BASE_PATH / cfg.skybox_path)
        if img_bgr is None:
            raise RuntimeError(f"Unable to load skybox: {cfg.skybox_path}")
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA) * 33
        del img_bgr
        skybox_gpu = cp.asarray(img_rgba.astype(np.float16))
        del img_rgba
        self.skybox_tex = create_texture_array_2d(skybox_gpu, 4, (1, 1, 1, 1), True)
        del skybox_gpu

        print(f"[GPU {self.gpu_id}] Loading disk texture and colour LUT...")
        prebaked_data = np.load(BASE_PATH / cfg.prebaked_disk_path)
        self.disk_tex = create_texture_array_3d(
            cp.asarray(prebaked_data, dtype=cp.float16), 4, (0, 0, 1, 1, 1), True
        )
        del prebaked_data

        lut_rgb = np.load(BASE_PATH / cfg.color_lut_path)
        lut_rgba = np.ones((*lut_rgb.shape[:-1], 4), dtype=lut_rgb.dtype)
        lut_rgba[..., :3] = lut_rgb
        lut_gpu = cp.asarray(lut_rgba, dtype=cp.float16)
        del lut_rgb, lut_rgba
        self.color_tex = create_texture_array_2d(lut_gpu, 4, (1, 1, 1, 1), True)
        del lut_gpu

        self._compile_kernels()
        self._create_frame_buffers()
        self._record_postprocess_graph()

    def _compile_kernels(self) -> None:
        kernel_file = BASE_PATH / cfg.kernel_path
        compile_opts = ["-use_fast_math", f"-I{BASE_PATH / 'krnls'}", "-lineinfo"]
        if cfg.USE_RK4:
            compile_opts.append("-DUSE_RK4")
        if cfg.NO_DEPTH_JITTER:
            compile_opts.append("-DNO_DEPTH_JITTER")
        if cfg.RAND_SAMP_DISK:
            compile_opts.append("-DRAND_SAMP_DISK")
        if cfg.NO_BKGD_DOPPLER:
            compile_opts.append("-DNO_BKGD_DOPPLER")

        trace_source = kernel_file.read_text(encoding="utf-8")
        trace_module = cp.RawModule(code=trace_source, options=tuple(compile_opts))
        self.trace_rays_kernel = trace_module.get_function("blackholekernel")

        bloom_file = BASE_PATH / cfg.bloom_kernel_path
        bloom_source = bloom_file.read_text(encoding="utf-8")
        bloom_opts = ["-use_fast_math", f"-I{BASE_PATH / 'krnls'}"]
        if cfg.USE_ACES:
            bloom_opts.append("-DUSE_ACES")
        if cfg.NOT_USE_S_CURVE:
            bloom_opts.append("-DNOT_USE_S_CURVE")
        bloom_module = cp.RawModule(code=bloom_source, options=tuple(bloom_opts))
        self.gauss_h = bloom_module.get_function("gaussianBlurH")
        self.gauss_w = bloom_module.get_function("gaussianBlurW")
        self.bloom = bloom_module.get_function("compositeBloom")
        self.extract_bright = bloom_module.get_function("extractBright")
        self.downsample = bloom_module.get_function("downsample2x")

    def _create_frame_buffers(self) -> None:
        target_size = 8
        self.num_levels = max(1, int(np.round(np.log2(min(self.w, self.h) / target_size))))
        self.down_resolutions: list[tuple[int, int]] = []
        self.down_texs = []
        self.down_surfs = []
        self.tmp_texs = []
        self.tmp_surfs = []

        curr_w, curr_h = self.w, self.h
        for _ in range(self.num_levels):
            curr_w, curr_h = max(1, curr_w // 2), max(1, curr_h // 2)
            buffer = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
            texture, surface = create_texture_surface_union_2d(buffer, 4, (1, 1, 1, 1))
            self.down_resolutions.append((curr_w, curr_h))
            self.down_texs.append(texture)
            self.down_surfs.append(surface)

            temp_buffer = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
            texture, surface = create_texture_surface_union_2d(temp_buffer, 4, (1, 1, 1, 1))
            self.tmp_texs.append(texture)
            self.tmp_surfs.append(surface)

        intermediate = cp.empty((self.h, self.w, 4), dtype=cp.float32)
        self.frame_inter_tex, self.frame_inter_surf = create_texture_surface_union_2d(
            intermediate, 4, (1, 1, 0, 1)
        )
        del intermediate
        self.current_frame = cp.empty(self.h * self.w * 4, dtype=cp.uint8)

        bright_buffer = cp.zeros((self.h, self.w, 4), dtype=cp.float32)
        self.bright_tex, self.bright_surf = create_texture_surface_union_2d(
            bright_buffer, 4, (1, 1, 0, 1)
        )
        del bright_buffer
        self.tex_ptrs = [texture.ptr for texture in self.down_texs]
        self.tex_group = cp.array(self.tex_ptrs, dtype=cp.uint64)

        camera = cfg.cam_pos_init.astype(np.float32)
        self.forward, self.right, self.up = self._camera_axes()
        self.static_kernel_args = (
            cp.uint64(self.frame_inter_surf.ptr),
            cp.uint64(self.skybox_tex.ptr),
            cp.uint64(self.disk_tex.ptr),
            cp.uint64(self.color_tex.ptr),
            cp.float32(cfg.start_t),
            cp.float32(camera[0]), cp.float32(camera[1]), cp.float32(camera[2]),
            cp.float32(self.forward[0]), cp.float32(self.forward[1]), cp.float32(self.forward[2]),
            cp.float32(self.right[0]), cp.float32(self.right[1]), cp.float32(self.right[2]),
            cp.float32(self.up[0]), cp.float32(self.up[1]), cp.float32(self.up[2]),
            cp.float32(cfg.vx), cp.float32(cfg.vy), cp.float32(cfg.vz),
            cp.int32(self.w), cp.int32(self.h),
            cp.float32(self.w * cfg.PIXEL_PITCH), cp.float32(self.h * cfg.PIXEL_PITCH),
            cp.float32(cfg.focal_length), cp.float32(cfg.step), cp.int32(cfg.maxstep),
            cp.int32(cfg.SSAA_COUNT), cp.int32(1),
        )
        self.render_stream = cp.cuda.Stream(non_blocking=True)

    @staticmethod
    def _camera_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array(
            [
                np.cos(cfg.cam_yaw) * np.cos(cfg.cam_pitch),
                np.sin(cfg.cam_yaw) * np.cos(cfg.cam_pitch),
                np.sin(cfg.cam_pitch),
            ],
            dtype=np.float32,
        )
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right0 = np.cross(forward, world_up)
        right0 /= np.linalg.norm(right0)
        up0 = np.cross(right0, forward)
        right = right0 * np.cos(cfg.cam_roll) + up0 * np.sin(cfg.cam_roll)
        up = up0 * np.cos(cfg.cam_roll) - right0 * np.sin(cfg.cam_roll)
        return forward, right, up

    def _record_postprocess_graph(self) -> None:
        threshold = np.float32(7)
        capture_stream = cp.cuda.Stream(non_blocking=True)
        with capture_stream:
            capture_stream.begin_capture()
            self.extract_bright(
                (self.grid_x, self.grid_y), (self.block_x, self.block_y),
                (cp.uint64(self.frame_inter_tex.ptr), cp.uint64(self.bright_surf.ptr),
                 cp.int32(self.w), cp.int32(self.h), cp.float32(1.0), cp.float32(threshold)),
            )
            prev_tex_ptr = self.bright_tex.ptr
            for index, (out_w, out_h) in enumerate(self.down_resolutions):
                grid_x = (out_w + self.block_x - 1) // self.block_x
                grid_y = (out_h + self.block_y - 1) // self.block_y
                self.downsample(
                    (grid_x, grid_y), (self.block_x, self.block_y),
                    (cp.uint64(prev_tex_ptr), cp.uint64(self.down_surfs[index].ptr),
                     cp.int32(out_w), cp.int32(out_h)),
                )
                self.gauss_h(
                    (grid_x, grid_y), (self.block_x, self.block_y),
                    (cp.uint64(self.tmp_surfs[index].ptr), cp.int32(out_w), cp.int32(out_h),
                     cp.uint64(self.tex_ptrs[index]), cp.float32(1.0)),
                )
                self.gauss_w(
                    (grid_x, grid_y), (self.block_x, self.block_y),
                    (cp.uint64(self.down_surfs[index].ptr), cp.int32(out_w), cp.int32(out_h),
                     cp.uint64(self.tmp_texs[index].ptr), cp.float32(1.0)),
                )
                prev_tex_ptr = self.tex_ptrs[index]
            self.bloom(
                (self.grid_x, self.grid_y), (self.block_x, self.block_y),
                (self.current_frame, cp.int32(self.w), cp.int32(self.h),
                 cp.uint64(self.frame_inter_tex.ptr), self.tex_group, cp.int32(self.num_levels),
                 cp.int32(self.w), cp.int32(self.h)),
            )
            self.postprocess_graph = capture_stream.end_capture()

    def render_frame(self, frame_idx: int) -> None:
        """Render an independent fixed-camera frame and write its PNG."""
        frame_time = np.float32(cfg.start_t + (frame_idx - 1) * self.time_step)
        args = list(self.static_kernel_args)
        args[4] = cp.float32(frame_time)
        args[-1] = cp.int32(frame_idx)

        started = time.perf_counter()
        with self.render_stream:
            self.trace_rays_kernel(
                (self.grid_x, self.grid_y), (self.block_x, self.block_y), tuple(args)
            )
            self.postprocess_graph.launch(stream=self.render_stream)
        self.render_stream.synchronize()

        rgba = cp.asnumpy(self.current_frame.reshape((self.h, self.w, 4)))
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        path = self.output_dir / f"frame_{frame_idx:05d}.png"
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write {path}")
        elapsed = time.perf_counter() - started
        print(
            f"[GPU {self.gpu_id}] frame {frame_idx:05d}/{cfg.total_frames:05d} "
            f"t={frame_time:.3f} {elapsed:.3f}s"
        )


class CpuRenderer:
    """One frame-parallel CPU renderer that mirrors the active GPU pipeline."""

    def __init__(self, cpu_cores: int, time_step: float, output_dir: Path):
        import render_cpu as cpu_render

        self.cpu = cpu_render
        self.cpu_cores = cpu_cores
        self.time_step = np.float32(time_step)
        self.output_dir = output_dir
        if not cpu_render.native_extension_exists():
            cpu_render.build_native_extension()
        self._initialize()

    @staticmethod
    def _as_gpu_half_texture(texture: np.ndarray) -> np.ndarray:
        """Match the float16 uploads performed by GpuRenderer."""

        return np.ascontiguousarray(np.asarray(texture, dtype=np.float16), dtype=np.float32)

    def _initialize(self) -> None:
        self.w, self.h = cfg.w, cfg.h
        print("[CPU] Loading textures...")
        img_bgr = read_image_bgr(BASE_PATH / cfg.skybox_path)
        if img_bgr is None:
            raise RuntimeError(f"Unable to load skybox: {cfg.skybox_path}")
        # Keep multiplication before float16 conversion, as in GpuRenderer.
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA) * 33
        del img_bgr
        self.skybox = self._as_gpu_half_texture(img_rgba)
        del img_rgba

        self.disk = self._as_gpu_half_texture(np.load(BASE_PATH / cfg.prebaked_disk_path, mmap_mode="r"))
        self.color_lut = self._as_gpu_half_texture(np.load(BASE_PATH / cfg.color_lut_path))

        self.forward, self.right, self.up = GpuRenderer._camera_axes()
        self.flags = self.cpu.RenderFlag(0)
        if cfg.USE_RK4:
            self.flags |= self.cpu.RenderFlag.USE_RK4
        if not cfg.NO_DEPTH_JITTER:
            self.flags |= self.cpu.RenderFlag.DEPTH_JITTER
        if cfg.RAND_SAMP_DISK:
            self.flags |= self.cpu.RenderFlag.RAND_SAMP_DISK
        # Deliberately do not enable OPACITY_CHANGE: GpuRenderer does not pass
        # -DOPACITY_CHANGE to the CUDA compiler, even when config exposes it.
        if not cfg.NO_BKGD_DOPPLER:
            self.flags |= self.cpu.RenderFlag.BACKGROUND_DOPPLER

        self.postprocess = self.cpu.PostprocessParams(
            bloom_threshold=7.0,
            use_aces=cfg.USE_ACES,
            not_use_s_curve=cfg.NOT_USE_S_CURVE,
        )

    def render_frame(self, frame_idx: int) -> None:
        frame_time = float(cfg.start_t + (frame_idx - 1) * self.time_step)
        params = self.cpu.RenderParams(
            time=frame_time,
            camera_position=cfg.cam_pos_init,
            forward=self.forward,
            right=self.right,
            up=self.up,
            velocity=(float(cfg.vx), float(cfg.vy), float(cfg.vz)),
            physwidth=self.w * cfg.PIXEL_PITCH,
            physheight=self.h * cfg.PIXEL_PITCH,
            focal_length=cfg.focal_length,
            step=cfg.step,
            maxstep=cfg.maxstep,
            jitternum=cfg.SSAA_COUNT,
            frames=frame_idx,
            flags=self.flags,
        )
        started = time.perf_counter()
        rgba = self.cpu.render_rgba8_new(
            self.w,
            self.h,
            self.skybox,
            self.disk,
            self.color_lut,
            params,
            postprocess=self.postprocess,
            tile_size=(64, 16),
            workers=self.cpu_cores,
        )
        path = self.output_dir / f"frame_{frame_idx:05d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)):
            raise RuntimeError(f"Failed to write {path}")
        elapsed = time.perf_counter() - started
        print(
            f"[CPU] frame {frame_idx:05d}/{cfg.total_frames:05d} "
            f"t={frame_time:.3f} {elapsed:.3f}s ({self.cpu_cores} cores)"
        )


def worker(gpu_id: int, scheduler: FrameScheduler, time_step: float, output_dir: Path) -> None:
    current_frame: int | None = None
    try:
        with cp.cuda.Device(gpu_id):
            renderer = GpuRenderer(gpu_id, time_step, output_dir)
            print(f"[GPU {gpu_id}] Renderer ready")
            while (current_frame := scheduler.claim()) is not None:
                try:
                    renderer.render_frame(current_frame)
                except Exception:
                    scheduler.fail(f"GPU {gpu_id}", current_frame, traceback.format_exc())
                    print(f"[GPU {gpu_id}] frame {current_frame:05d} failed")
                    break
                else:
                    scheduler.finish(current_frame)
                    current_frame = None
                    gc.collect()
    except Exception:
        if current_frame is not None:
            scheduler.fail(f"GPU {gpu_id}", current_frame, traceback.format_exc())
        else:
            scheduler.record_worker_error(f"GPU {gpu_id}", traceback.format_exc())
        print(f"[GPU {gpu_id}] worker stopped during initialization")


def cpu_worker(cpu_cores: int, scheduler: FrameScheduler, time_step: float, output_dir: Path) -> None:
    """Claim frames alongside GPU workers, using one CPU renderer instance."""

    current_frame: int | None = None
    try:
        renderer = CpuRenderer(cpu_cores, time_step, output_dir)
        print(f"[CPU] Renderer ready with {cpu_cores} cores")
        while (current_frame := scheduler.claim()) is not None:
            try:
                renderer.render_frame(current_frame)
            except Exception:
                scheduler.fail("CPU", current_frame, traceback.format_exc())
                print(f"[CPU] frame {current_frame:05d} failed")
                break
            else:
                scheduler.finish(current_frame)
                current_frame = None
                gc.collect()
    except Exception:
        if current_frame is not None:
            scheduler.fail("CPU", current_frame, traceback.format_exc())
        else:
            scheduler.record_worker_error("CPU", traceback.format_exc())
        print("[CPU] worker stopped during initialization")


def parse_gpu_ids(value: str, gpu_count: int) -> list[int]:
    if value == "all":
        return list(range(gpu_count))
    gpu_ids = [int(part) for part in value.split(",") if part.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must be 'all' or a comma-separated list of unique IDs")
    if any(gpu_id < 0 or gpu_id >= gpu_count for gpu_id in gpu_ids):
        raise ValueError(f"GPU IDs must be in [0, {gpu_count - 1}]")
    return gpu_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-camera multi-GPU frame renderer")
    parser.add_argument("--gpus", default="all", help="GPU IDs, for example: 0,1. Default: all")
    parser.add_argument("--time-step", type=float, default=float(cfg.d_tau), help="Scene time increment per frame")
    parser.add_argument("--overwrite", action="store_true", help="Render frames even if their PNG already exists")
    parser.add_argument("--use-cpu", action="store_true", help="Add one CPU frame worker alongside the GPU workers")
    parser.add_argument("--cpu-cores", type=int, metavar="N", help="CPU cores assigned to the one CPU worker")
    args = parser.parse_args()

    if args.time_step <= 0:
        parser.error("--time-step must be positive")
    if args.use_cpu and args.cpu_cores is None:
        parser.error("--use-cpu requires --cpu-cores N")
    if not args.use_cpu and args.cpu_cores is not None:
        parser.error("--cpu-cores requires --use-cpu")
    if args.cpu_cores is not None:
        available_cores = os.cpu_count() or 1
        if args.cpu_cores < 1 or args.cpu_cores > available_cores:
            parser.error(f"--cpu-cores must be in [1, {available_cores}]")

    if cp is None:
        raise RuntimeError("CuPy is required to run render_hybrid.py")
    gpu_count = cp.cuda.runtime.getDeviceCount()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU is visible to CuPy")
    gpu_ids = parse_gpu_ids(args.gpus, gpu_count)

    output_dir = BASE_PATH / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(BASE_PATH / ".cupy_cache"))
    scheduler = FrameScheduler(cfg.total_frames, output_dir, args.overwrite)
    if getattr(cfg, "OPACITY_CHANGE", False):
        print("Note: OPACITY_CHANGE is not passed to the CUDA kernel; existing output is preserved.")
    manifest_path = start_manifest(
        output_dir,
        "hybrid",
        cfg,
        {
            "arguments": vars(args),
            "gpu_ids": gpu_ids,
            "cpu_accelerator_cores": args.cpu_cores if args.use_cpu else 0,
        },
    )
    cpu_description = f"; CPU worker={args.cpu_cores} cores" if args.use_cpu else ""
    print(f"Using GPUs {gpu_ids}{cpu_description}; {scheduler.summary()}; time step={args.time_step}")

    threads = [
        threading.Thread(
            target=worker,
            name=f"gpu-{gpu_id}",
            args=(gpu_id, scheduler, args.time_step, output_dir),
        )
        for gpu_id in gpu_ids
    ]
    if args.use_cpu:
        threads.append(
            threading.Thread(
                target=cpu_worker,
                name="cpu",
                args=(args.cpu_cores, scheduler, args.time_step, output_dir),
            )
        )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = scheduler.summary()
    print(f"Render summary: {summary}")
    finish_manifest(
        manifest_path,
        "failed" if scheduler.errors else "complete",
        {"summary": summary},
    )
    if scheduler.errors:
        for worker_name, frame_idx, error in scheduler.errors:
            print(f"\n{worker_name}, frame {frame_idx:05d} error:\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
