"""CPU accelerator for the black-hole ray tracer and post-processing graph.

Build the local pybind11 extension once with ``python tools/build_cpu_render.py``.
The native module accepts NumPy arrays directly: background ``(H, W, 3|4)``,
disk LUT ``(R, Z, Phi, 3|4)``, and colour LUT ``(H, W, 3|4)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from types import ModuleType
from typing import Sequence

import numpy as np


class RenderFlag(IntFlag):
    """Runtime equivalents of the optional CUDA preprocessor defines."""

    USE_RK4 = 1 << 0
    DEPTH_JITTER = 1 << 1
    RAND_SAMP_DISK = 1 << 2
    OPACITY_CHANGE = 1 << 3
    BACKGROUND_DOPPLER = 1 << 4
    DISK_DOPPLER_FOLLOW_BACKGROUND = 1 << 5
    PHOTON_RING_OPTIMIZATION = 1 << 6


@dataclass(frozen=True)
class PostprocessParams:
    """The runtime values passed to ``krnls/bloom.cu``."""

    bloom_threshold: float = 12.0
    use_aces: bool = True
    not_use_s_curve: bool = False


@dataclass(frozen=True)
class RenderParams:
    time: float
    camera_position: Sequence[float]
    forward: Sequence[float]
    right: Sequence[float]
    up: Sequence[float]
    velocity: Sequence[float] = (0.0, 0.0, 0.0)
    physwidth: float = 3.2
    physheight: float = 2.0
    focal_length: float = 4.0
    step: float = 0.02
    maxstep: int = 4000
    jitternum: int = 2
    frames: int = 1
    # Defaults match the source file with no CUDA preprocessor flags set.
    flags: RenderFlag = RenderFlag.DEPTH_JITTER | RenderFlag.BACKGROUND_DOPPLER

    def _native(self) -> object:
        def vector(name: str, value: Sequence[float]) -> tuple[float, float, float]:
            if len(value) != 3:
                raise ValueError(f"{name} must contain exactly three values")
            return tuple(float(component) for component in value)  # type: ignore[return-value]

        camera = vector("camera_position", self.camera_position)
        forward = vector("forward", self.forward)
        right = vector("right", self.right)
        up = vector("up", self.up)
        velocity = vector("velocity", self.velocity)
        if self.maxstep <= 0 or self.jitternum <= 0 or self.step <= 0.0 or self.focal_length <= 0.0:
            raise ValueError("step, focal_length, maxstep, and jitternum must be positive")
        if sum(component * component for component in velocity) >= 1.0:
            raise ValueError("velocity magnitude must be smaller than the speed of light")
        native = _native_module().RenderParams()
        native.time = self.time
        native.cam_pos_x, native.cam_pos_y, native.cam_pos_z = camera
        native.fwd_x, native.fwd_y, native.fwd_z = forward
        native.right_x, native.right_y, native.right_z = right
        native.up_x, native.up_y, native.up_z = up
        native.vfwd, native.vright, native.vup = velocity
        native.physwidth = self.physwidth
        native.physheight = self.physheight
        native.focal_length = self.focal_length
        native.step = self.step
        native.maxstep = self.maxstep
        native.jitternum = self.jitternum
        native.frames = self.frames
        native.flags = int(self.flags)
        return native


_ROOT = Path(__file__).resolve().parent
_EXTENSION_PATH = _ROOT / "build" / f"cpu_render_native{sysconfig.get_config_var('EXT_SUFFIX')}"
_NATIVE_MODULE: ModuleType | None = None


def build_native_extension() -> Path:
    """Compile ``cpu/cpu_render.cpp`` into the pybind11 extension consumed by this module."""

    subprocess.run([sys.executable, str(_ROOT / "tools" / "build_cpu_render.py")], cwd=_ROOT, check=True)
    importlib.invalidate_caches()
    return _EXTENSION_PATH


def native_extension_exists() -> bool:
    """Return whether the pybind11 ``cpu_render_native`` module is built."""

    return _EXTENSION_PATH.is_file()


def _native_module() -> ModuleType:
    global _NATIVE_MODULE
    if _NATIVE_MODULE is None:
        if not _EXTENSION_PATH.exists():
            raise RuntimeError(
                f"{_EXTENSION_PATH.name} has not been built. Run `python tools/build_cpu_render.py` "
                "or call render_cpu.build_native_extension() first."
            )
        module_name = "cpu_render_native"
        spec = importlib.util.spec_from_file_location(module_name, _EXTENSION_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load pybind11 extension: {_EXTENSION_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _NATIVE_MODULE = module
    return _NATIVE_MODULE


def _tile_dimensions(tile_size: tuple[int, int]) -> tuple[int, int]:
    if len(tile_size) != 2 or tile_size[0] <= 0 or tile_size[1] <= 0:
        raise ValueError("tile_size must contain two positive integers")
    return int(tile_size[0]), int(tile_size[1])


def render(
    raw_img: np.ndarray,
    background: np.ndarray,
    prebaked_disk: np.ndarray,
    color_lut: np.ndarray,
    params: RenderParams,
    *,
    tile_size: tuple[int, int] = (32, 8),
    workers: int | None = None,
) -> np.ndarray:
    """Render in-place into a ``(height, width, 4)`` float32 NumPy array.

    ``workers=None`` uses the machine's hardware-thread count.  The call
    returns ``raw_img`` for convenient chaining.
    """

    tile_width, tile_height = _tile_dimensions(tile_size)
    _native_module().render_into(
        raw_img,
        background,
        prebaked_disk,
        color_lut,
        params._native(),
        tile_width,
        tile_height,
        0 if workers is None else int(workers),
    )
    return raw_img


def render_new(
    width: int,
    height: int,
    background: np.ndarray,
    prebaked_disk: np.ndarray,
    color_lut: np.ndarray,
    params: RenderParams,
    *,
    tile_size: tuple[int, int] = (32, 8),
    workers: int | None = None,
) -> np.ndarray:
    """Allocate a float32 RGBA buffer, render into it, and return it."""

    tile_width, tile_height = _tile_dimensions(tile_size)
    return _native_module().render_new(
        int(width), int(height), background, prebaked_disk, color_lut, params._native(), tile_width, tile_height,
        0 if workers is None else int(workers),
    )


def render_to_rgba8(
    output_img: np.ndarray,
    background: np.ndarray,
    prebaked_disk: np.ndarray,
    color_lut: np.ndarray,
    params: RenderParams,
    *,
    postprocess: PostprocessParams = PostprocessParams(),
    tile_size: tuple[int, int] = (32, 8),
    workers: int | None = None,
) -> np.ndarray:
    """Run the complete CPU pipeline and write final RGBA8 pixels in-place.

    The native call owns all HDR, bright-pass, downsample and blur buffers.
    ``output_img`` is the only framebuffer visible to Python, with shape
    ``(height, width, 4)`` and ``uint8`` dtype.  pybind11 releases the Python
    GIL for this complete call.
    """

    tile_width, tile_height = _tile_dimensions(tile_size)
    _native_module().render_rgba8_into(
        output_img,
        background,
        prebaked_disk,
        color_lut,
        params._native(),
        tile_width,
        tile_height,
        0 if workers is None else int(workers),
        float(postprocess.bloom_threshold),
        bool(postprocess.use_aces),
        bool(postprocess.not_use_s_curve),
    )
    return output_img


def render_rgba8_new(
    width: int,
    height: int,
    background: np.ndarray,
    prebaked_disk: np.ndarray,
    color_lut: np.ndarray,
    params: RenderParams,
    *,
    postprocess: PostprocessParams = PostprocessParams(),
    tile_size: tuple[int, int] = (32, 8),
    workers: int | None = None,
) -> np.ndarray:
    """Allocate the final uint8 RGBA image and run :func:`render_to_rgba8`."""

    tile_width, tile_height = _tile_dimensions(tile_size)
    return _native_module().render_rgba8_new(
        int(width),
        int(height),
        background,
        prebaked_disk,
        color_lut,
        params._native(),
        tile_width,
        tile_height,
        0 if workers is None else int(workers),
        float(postprocess.bloom_threshold),
        bool(postprocess.use_aces),
        bool(postprocess.not_use_s_curve),
    )


CPU_TEST_WORKERS = 24
CPU_TEST_TILE_SIZE = (64, 16)
CPU_TEST_OUTPUT_DIR = "output_frames_cpu_offline_config_test"
# Keep the configured 4096x2160 frame, but use one ray sample per pixel for
# a practical CPU output test.
CPU_TEST_SSAA = 16


def _camera_vectors(yaw: float, pitch: float, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same camera-basis construction as render_single.py."""

    world_up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    forward = np.array(
        (np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)), dtype=np.float32
    )
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    right, up = right * np.cos(roll) + up * np.sin(roll), up * np.cos(roll) - right * np.sin(roll)
    return forward, right, up


def _load_background_rgba(path: Path) -> np.ndarray:
    """Load the configured skybox exactly as render_single.py does."""

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("The CPU image test needs opencv-python (`pip install opencv-python`).") from error

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read the configured skybox: {path}")
    if image.ndim == 2:
        rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported skybox channel count: {image.shape}")
    # In the GPU script this multiplication occurs while rgba is uint8, then
    # the result is uploaded as float16. Keep both conversion steps here so
    # the CPU sees the same texture values.
    scaled = np.multiply(rgba, np.uint8(33), dtype=np.uint8)
    return _as_gpu_half_texture(scaled)


def _as_gpu_half_texture(texture: np.ndarray) -> np.ndarray:
    """Match the float16 texture uploads in render_single.py."""

    return np.ascontiguousarray(np.asarray(texture, dtype=np.float16), dtype=np.float32)


def run_full_resolution_low_ssaa_test() -> Path:
    """Render and save one full-resolution cfg/offline.py frame at low SSAA."""

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("The CPU image test needs opencv-python (`pip install opencv-python`).") from error
    if not _EXTENSION_PATH.exists():
        build_native_extension()

    from cfg import offline as config

    test_width = int(config.w)
    test_height = int(config.h)
    background = _load_background_rgba(_ROOT / config.skybox_path)
    prebaked_disk = _as_gpu_half_texture(np.load(_ROOT / config.prebaked_disk_path, mmap_mode="r"))
    color_lut = _as_gpu_half_texture(np.load(_ROOT / config.color_lut_path))
    forward, right, up = _camera_vectors(config.cam_yaw, config.cam_pitch, config.cam_roll)

    flags = RenderFlag(0)
    if getattr(config, "USE_RK4", False):
        flags |= RenderFlag.USE_RK4
    if not getattr(config, "NO_DEPTH_JITTER", False):
        flags |= RenderFlag.DEPTH_JITTER
    if getattr(config, "RAND_SAMP_DISK", False):
        flags |= RenderFlag.RAND_SAMP_DISK
    # cfg/offline.py exposes OPACITY_CHANGE, but render_single.py
    # never appends -DOPACITY_CHANGE to compile_opts.  The GPU therefore uses
    # the ordinary 1.7x opacity branch; keep the CPU path identical.
    if not getattr(config, "NO_BKGD_DOPPLER", False):
        flags |= RenderFlag.BACKGROUND_DOPPLER

    params = RenderParams(
        time=float(config.start_t),
        camera_position=config.cam_pos_init,
        forward=forward,
        right=right,
        up=up,
        velocity=(float(config.vx), float(config.vy), float(config.vz)),
        physwidth=test_width * float(config.PIXEL_PITCH),
        physheight=test_height * float(config.PIXEL_PITCH),
        focal_length=float(config.focal_length),
        step=float(config.step),
        maxstep=int(config.maxstep),
        jitternum=CPU_TEST_SSAA,
        frames=1,
        flags=flags,
    )

    started = time.perf_counter()
    postprocess = PostprocessParams(
        bloom_threshold=12.0,
        use_aces=bool(getattr(config, "USE_ACES", False)),
        not_use_s_curve=bool(getattr(config, "NOT_USE_S_CURVE", False)),
    )
    frame = render_rgba8_new(
        test_width,
        test_height,
        background,
        prebaked_disk,
        color_lut,
        params,
        tile_size=CPU_TEST_TILE_SIZE,
        workers=CPU_TEST_WORKERS,
        postprocess=postprocess,
    )
    elapsed_seconds = time.perf_counter() - started

    output_dir = _ROOT / CPU_TEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"cpu_fullres_{test_width}x{test_height}_ssaa{params.jitternum}_{stamp}"
    image_path = output_dir / f"{stem}.png"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)):
        raise RuntimeError(f"Unable to write test image: {image_path}")

    report = {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "resolution": [test_width, test_height],
        "ssaa": params.jitternum,
        "render_mode": "full-resolution low-SSAA",
        "workers": CPU_TEST_WORKERS,
        "tile_size": list(CPU_TEST_TILE_SIZE),
        "maxstep": params.maxstep,
        "bloom_threshold": postprocess.bloom_threshold,
        "config": "cfg/offline.py",
        "image": image_path.name,
    }
    report_path = output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"CPU full-resolution render: {test_width}x{test_height}, SSAA {params.jitternum}, "
        f"{elapsed_seconds:.3f}s, workers={CPU_TEST_WORKERS}; saved to {image_path}"
    )
    return image_path


# Compatibility name for callers that used the old test entry point.
run_configured_test = run_full_resolution_low_ssaa_test


if __name__ == "__main__":
    run_full_resolution_low_ssaa_test()
