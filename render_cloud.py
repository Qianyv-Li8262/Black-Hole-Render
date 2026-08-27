"""Offline entry point for the cloud-volume black hole experiment.

This is intentionally self-contained: paths, camera, render settings, and compile
flags are hardcoded near the top so experiments can be saved by copying this file.
"""

from __future__ import annotations

import gc
import os
import time

import cv2
import cupy as cp
import numpy as np

from helpers.cuda_tex import create_texture_array_2d, create_texture_array_3d, create_texture_surface_union_2d
from helpers.render_manifest import finish_manifest, start_manifest


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

# Resolution / sampling. Start conservative; cloud scattering is much heavier
# than the old purely emissive disk path.
W, H = 1600, 900
TOTAL_FRAMES = 1
SSAA_COUNT = 64
BLOCK_X, BLOCK_Y = 32, 8
PIXEL_PITCH = 0.002

# Camera and motion parameters.
CAM_POS_INIT = np.array([0,0,80], dtype=np.float32)
CAM_YAW = -10.4
CAM_PITCH = -1.57
CAM_ROLL = 0.0
FOCAL_LENGTH = 2.0
VX, VY, VZ = 0,0,0
START_T = 20.5
FRAME_DT = 0.5

# Ray marching.
STEP = 0.04
MAXSTEP = 4000

# Files, relative to this script.
SKYBOX_PATH = "assets/white.bmp"
CLOUD_DENSITY_PATH = "cache/accretion_disk_lut.npy"
COLOR_LUT_PATH = "cache/color_lut2.npy"
KERNEL_PATH = "krnls/cloud.cu"
BLOOM_KERNEL_PATH = "krnls/bloom.cu"
OUTPUT_DIR = "output_frames_cloud_experiment"

# Texture exposure / storage.
SKY_EXPOSURE = 1/128
USE_HALF_TEXTURES = True

# Compile flags.
NO_DEPTH_JITTER = True
RAND_SAMP_DISK = True
NO_BKGD_DOPPLER = False
USE_ACES = False
NOT_USE_S_CURVE = False

# Bloom controls.
BLOOM_THRESHOLD = np.float32(1.7)
BLUR_SCALE = np.float32(1.0)
BLOOM_TARGET_SIZE = 8

# Debug aid: synchronize after trace before launching bloom. If an illegal
# address remains, this tells whether it comes from ray marching or postprocess.
SYNC_AFTER_TRACE = False


# -----------------------------------------------------------------------------
# Resource loading
# -----------------------------------------------------------------------------


def load_sky_texture(base_path: str):
    path = os.path.join(base_path, SKYBOX_PATH)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load skybox: {path}")

    if img.ndim == 2:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported skybox shape: {img.shape}")

    img_rgba = img_rgba.astype(np.float32) * SKY_EXPOSURE
    sky_gpu = cp.asarray(img_rgba, dtype=cp.float16 if USE_HALF_TEXTURES else cp.float32)
    return create_texture_array_2d(sky_gpu, 4, (1, 1, 1, 1), USE_HALF_TEXTURES)


def load_cloud_texture(base_path: str):
    path = os.path.join(base_path, CLOUD_DENSITY_PATH)
    data = np.load(path)
    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"Cloud texture must have shape (r,z,phi,4), got {data.shape}")

    # Layout is (r, z, phi, rgba). cuda_tex maps this so tex3D x=phi, y=z, z=r.
    data_gpu = cp.asarray(data, dtype=cp.float16 if USE_HALF_TEXTURES else cp.float32)
    return create_texture_array_3d(data_gpu, 4, (0, 1, 1, 1, 1), USE_HALF_TEXTURES), data.shape


def load_color_lut(base_path: str):
    path = os.path.join(base_path, COLOR_LUT_PATH)
    lut_rgb = np.load(path)
    lut_rgba = np.ones((*lut_rgb.shape[:-1], 4), dtype=lut_rgb.dtype)
    lut_rgba[..., :3] = lut_rgb
    lut_gpu = cp.asarray(lut_rgba, dtype=cp.float16 if USE_HALF_TEXTURES else cp.float32)
    return create_texture_array_2d(lut_gpu, 4, (1, 1, 1, 1), USE_HALF_TEXTURES)


def compile_trace_kernel(base_path: str):
    path = os.path.join(base_path, KERNEL_PATH)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    opts = ["-use_fast_math", f"-I{os.path.join(base_path, 'krnls')}", "-lineinfo"]
    if NO_DEPTH_JITTER:
        opts.append("-DNO_DEPTH_JITTER")
    if RAND_SAMP_DISK:
        opts.append("-DRAND_SAMP_DISK")
    if NO_BKGD_DOPPLER:
        opts.append("-DNO_BKGD_DOPPLER")

    module = cp.RawModule(code=source, options=tuple(opts))
    return module, module.get_function("blackholekernel")


def compile_bloom_kernels(base_path: str):
    path = os.path.join(base_path, BLOOM_KERNEL_PATH)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    opts = ["-use_fast_math", f"-I{os.path.join(base_path, 'krnls')}"]
    if USE_ACES:
        opts.append("-DUSE_ACES")
    if NOT_USE_S_CURVE:
        opts.append("-DNOT_USE_S_CURVE")

    module = cp.RawModule(code=source, options=tuple(opts))
    return {
        "_module": module,
        "gauss_h": module.get_function("gaussianBlurH"),
        "gauss_w": module.get_function("gaussianBlurW"),
        "composite": module.get_function("compositeBloom"),
        "bright": module.get_function("extractBright"),
        "downsample": module.get_function("downsample2x"),
    }


# -----------------------------------------------------------------------------
# Math / setup helpers
# -----------------------------------------------------------------------------


def make_camera_basis(yaw: float, pitch: float, roll: float):
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    fwd = np.array(
        [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)],
        dtype=np.float32,
    )
    fwd /= np.linalg.norm(fwd)

    right0 = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right0)
    if right_norm < 1e-6:
        right0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        right0 /= right_norm

    up0 = np.cross(right0, fwd)
    up0 /= np.linalg.norm(up0)

    right = right0 * np.cos(roll) + up0 * np.sin(roll)
    up = up0 * np.cos(roll) - right0 * np.sin(roll)
    return fwd.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def make_surface(width: int, height: int):
    hdr = cp.empty((height, width, 4), dtype=cp.float32)
    tex, surf = create_texture_surface_union_2d(hdr, 4, (1, 1, 0, 1))
    del hdr
    return tex, surf


def setup_bloom(width: int, height: int, bloom_kernels):
    target_size = BLOOM_TARGET_SIZE
    num_levels = int(np.round(np.log2(min(width, height) / target_size)))
    num_levels = max(1, num_levels)

    frame_tex, frame_surf = make_surface(width, height)
    output_u8 = cp.empty((height * width * 4), dtype=cp.uint8)

    bright_buf = cp.zeros((height, width, 4), dtype=cp.float32)
    bright_tex, bright_surf = create_texture_surface_union_2d(bright_buf, 4, (1, 1, 0, 1))
    del bright_buf

    down_texs = []
    down_surfs = []
    tmp_texs = []
    tmp_surfs = []
    down_resolutions = []

    curr_w, curr_h = width, height
    for _ in range(num_levels):
        curr_w = max(1, curr_w // 2)
        curr_h = max(1, curr_h // 2)
        down_buf = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
        down_tex, down_surf = create_texture_surface_union_2d(down_buf, 4, (1, 1, 1, 1))
        tmp_buf = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
        tmp_tex, tmp_surf = create_texture_surface_union_2d(tmp_buf, 4, (1, 1, 1, 1))

        down_resolutions.append((curr_w, curr_h))
        down_texs.append(down_tex)
        down_surfs.append(down_surf)
        tmp_texs.append(tmp_tex)
        tmp_surfs.append(tmp_surf)

    block = (BLOCK_X, BLOCK_Y)
    grid = ((width + BLOCK_X - 1) // BLOCK_X, (height + BLOCK_Y - 1) // BLOCK_Y)
    tex_ptrs = [tex.ptr for tex in down_texs]
    tex_group = cp.array(tex_ptrs, dtype=cp.uint64)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()

        bloom_kernels["bright"](
            grid,
            block,
            (
                cp.uint64(frame_tex.ptr),
                cp.uint64(bright_surf.ptr),
                cp.int32(width),
                cp.int32(height),
                cp.float32(1.0),
                BLOOM_THRESHOLD,
            ),
        )

        prev_tex_ptr = bright_tex.ptr
        for i, (out_w, out_h) in enumerate(down_resolutions):
            out_grid = ((out_w + BLOCK_X - 1) // BLOCK_X, (out_h + BLOCK_Y - 1) // BLOCK_Y)
            bloom_kernels["downsample"](
                out_grid,
                block,
                (cp.uint64(prev_tex_ptr), cp.uint64(down_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h)),
            )
            bloom_kernels["gauss_h"](
                out_grid,
                block,
                (cp.uint64(tmp_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h), cp.uint64(tex_ptrs[i]), BLUR_SCALE),
            )
            bloom_kernels["gauss_w"](
                out_grid,
                block,
                (cp.uint64(down_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h), cp.uint64(tmp_texs[i].ptr), BLUR_SCALE),
            )
            prev_tex_ptr = tex_ptrs[i]

        bloom_kernels["composite"](
            grid,
            block,
            (
                output_u8,
                cp.int32(width),
                cp.int32(height),
                cp.uint64(frame_tex.ptr),
                tex_group,
                cp.int32(num_levels),
                cp.int32(width),
                cp.int32(height),
            ),
        )

        graph = stream.end_capture()

    return {
        "frame_tex": frame_tex,
        "frame_surf": frame_surf,
        "output_u8": output_u8,
        "graph": graph,
        "grid": grid,
        "block": block,
        "num_levels": num_levels,
        # Keep every object whose raw pointer/handle was captured by the graph.
        "bright_tex": bright_tex,
        "bright_surf": bright_surf,
        "down_texs": down_texs,
        "down_surfs": down_surfs,
        "tmp_texs": tmp_texs,
        "tmp_surfs": tmp_surfs,
        "tex_group": tex_group,
    }


def save_output(output_u8, frame_idx: int, output_dir: str):
    img_rgba = output_u8.reshape((H, W, 4))
    img_np = cp.asnumpy(img_rgba)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    out_path = os.path.join(output_dir, f"cloud_frame_{frame_idx:05d}.png")
    if not cv2.imwrite(out_path, img_bgr):
        raise RuntimeError(f"Failed to write {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Main render path
# -----------------------------------------------------------------------------


def main() -> None:
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = start_manifest(output_dir, "cloud", globals())

    print("Loading skybox texture...")
    sky_tex = load_sky_texture(base_path)

    print("Loading cloud density texture...")
    cloud_tex, cloud_shape = load_cloud_texture(base_path)
    print(f"  cloud shape: {cloud_shape}")

    print("Loading color LUT...")
    color_tex = load_color_lut(base_path)

    print("Compiling trace kernel...")
    trace_module, trace_kernel = compile_trace_kernel(base_path)

    print("Compiling bloom kernels...")
    bloom_kernels = compile_bloom_kernels(base_path)

    print("Preparing bloom graph...")
    bloom_state = setup_bloom(W, H, bloom_kernels)
    print(f"  bloom levels: {bloom_state['num_levels']}")

    fwd, right, up = make_camera_basis(CAM_YAW, CAM_PITCH, CAM_ROLL)

    grid = bloom_state["grid"]
    block = bloom_state["block"]
    render_stream = cp.cuda.Stream(non_blocking=True)

    print(f"Starting cloud experiment render: {TOTAL_FRAMES} frame(s)")
    for frame_idx in range(1, TOTAL_FRAMES + 1):
        started = time.time()
        t_val = START_T + (frame_idx - 1) * FRAME_DT

        kernel_args = (
            cp.uint64(bloom_state["frame_surf"].ptr),
            cp.uint64(sky_tex.ptr),
            cp.uint64(cloud_tex.ptr),
            cp.uint64(color_tex.ptr),
            cp.float32(t_val),
            cp.float32(CAM_POS_INIT[0]),
            cp.float32(CAM_POS_INIT[1]),
            cp.float32(CAM_POS_INIT[2]),
            cp.float32(fwd[0]),
            cp.float32(fwd[1]),
            cp.float32(fwd[2]),
            cp.float32(right[0]),
            cp.float32(right[1]),
            cp.float32(right[2]),
            cp.float32(up[0]),
            cp.float32(up[1]),
            cp.float32(up[2]),
            cp.float32(VX),
            cp.float32(VY),
            cp.float32(VZ),
            cp.int32(W),
            cp.int32(H),
            cp.float32(W * PIXEL_PITCH),
            cp.float32(H * PIXEL_PITCH),
            cp.float32(FOCAL_LENGTH),
            cp.float32(STEP),
            cp.int32(MAXSTEP),
            cp.int32(SSAA_COUNT),
            cp.int32(frame_idx),
        )

        with render_stream:
            trace_kernel(grid, block, kernel_args)
            if SYNC_AFTER_TRACE:
                render_stream.synchronize()
            bloom_state["graph"].launch(stream=render_stream)
        render_stream.synchronize()

        out_path = save_output(bloom_state["output_u8"], frame_idx, output_dir)
        elapsed = time.time() - started
        print(f"[Frame {frame_idx:05d}/{TOTAL_FRAMES:05d}] {elapsed:.3f} s -> {out_path}")
        gc.collect()

    print("Cloud experiment render complete.")
    finish_manifest(manifest_path, "complete", {"frames_rendered": TOTAL_FRAMES})
    _ = trace_module


if __name__ == "__main__":
    main()
