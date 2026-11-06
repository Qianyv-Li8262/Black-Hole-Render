# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
import argparse


def parse_render_args(argv=None):
    parser = argparse.ArgumentParser(description='Render from cfg initial conditions with a chosen output frame number.')
    parser.add_argument('--start', type=int, default=1, metavar='FRAME',
                        help='First saved frame number (default: 1); does not advance the camera')
    args = parser.parse_args(argv)
    if args.start < 1:
        parser.error('--start must be a positive integer')
    return args


# Embedded preview tools supply their own argument list instead of their CLI.
render_args = parse_render_args(globals().get('_render_argv'))
start_frame = render_args.start

import numpy as np
import cupy as cp
import cv2
import time
from helpers.cuda_tex import *
from helpers.disk_cache import load_disk_cache
import os, sys, gc, json
from cfg import offline as cfg
from cfg.offline import *
from helpers.render_manifest import finish_manifest, start_manifest
from helpers.camera_motion import FreeFallCamera, camera_basis


def update_after_frame(camera, frame_args, frame_idx, d_tau):
    """Advance free fall and pack camera/time arguments for the next frame.

    vx/vy/vz are initial local beta along forward/right/up. d_tau is the
    proper-time interval; scene time follows dt/dtau along the geodesic.
    Keep trajectory changes in this function.
    """
    if not camera.advance(d_tau):
        return False
    frame_args[4] = np.float32(camera.time)
    frame_args[5:8] = [np.float32(x) for x in camera.position]
    frame_args[17:20] = [np.float32(v) for v in camera.velocity]
    frame_args[-1] = np.int32(frame_idx + 1)
    return True


if not np.isfinite(d_tau) or d_tau <= 0:
    raise ValueError('cfg.d_tau must be positive and finite')
fwd, right, up = camera_basis(cam_yaw, cam_pitch, cam_roll)
camera = FreeFallCamera(cam_pos_init, (vx, vy, vz), (fwd, right, up), start_t)

base_path = os.path.dirname(os.path.abspath(__file__))


print('正在加载天空盒...')
img_bgr = cv2.imread(os.path.join(base_path, skybox_path),
                     cv2.IMREAD_UNCHANGED)
# img_bgr = cv2.imread(os.path.join(base_path, 'assets', 'black.bmp'),
#                      cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    print(f"错误:无法加载天空盒!")
    exit()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)*getattr(cfg, 'SKYBOX_SCALE', 330.0)
del img_bgr
img_float = cp.asarray(img_rgb.astype(np.float16))
del img_rgb
tex_handle = create_texture_array_2d(img_float,4,(1,1,1,1),True)
del img_float
print('  天空盒纹理就绪')


print('正在加载预烘焙吸积盘纹理...')
prebaked_data = load_disk_cache(os.path.join(base_path, prebaked_disk_path), getattr(cfg,'DISK_COEFFICIENTS',False))
ishalf=not getattr(cfg,'DISK_COEFFICIENTS',False)
tex_prebaked = create_texture_array_3d(cp.asarray(prebaked_data, dtype=cp.float16 if ishalf else cp.float32),4,(0,0,1,1,1),ishalf)
print(f"  数据 shape: {prebaked_data.shape}  dtype: {'half' if ishalf else 'float32'}")
del prebaked_data
print('  吸积盘 3D 纹理就绪')


print('正在加载颜色 LUT...')
lut_rgb = np.load(os.path.join(base_path, color_lut_path))
lut_rgba = np.ones((*lut_rgb.shape[:-1], 4), dtype=lut_rgb.dtype)
lut_rgba[..., :3] = lut_rgb
img_float = cp.asarray(lut_rgba, dtype=cp.float16)
tex_handle_color = create_texture_array_2d(img_float, 4, (1, 1, 1, 1), True)
print('  颜色 LUT 就绪')


print('正在编译 CUDA kernel...')
_trace_src = os.path.join(base_path, kernel_path)
with open(_trace_src, "r", encoding="utf-8") as f:
    cuda_source = f.read()
compile_opts = ['-use_fast_math', f'-I{os.path.join(base_path, "krnls")}', '-lineinfo']
if USE_RK4:
    compile_opts.append('-DUSE_RK4')
if NO_DEPTH_JITTER:
    compile_opts.append('-DNO_DEPTH_JITTER')
if RAND_SAMP_DISK:
    compile_opts.append('-DRAND_SAMP_DISK')
if NO_BKGD_DOPPLER:
    compile_opts.append('-DNO_BKGD_DOPPLER')
if getattr(cfg, 'OPACITY_CHANGE', False):
    compile_opts.append('-DOPACITY_CHANGE')
if getattr(cfg,'DISK_COEFFICIENTS',False):
    compile_opts.append('-DDISK_COEFFICIENTS')
module = cp.RawModule(code=cuda_source, options=tuple(compile_opts))
trace_rays_kernel = module.get_function("blackholekernel")


bloom_path = os.path.join(base_path, bloom_kernel_path)
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_opts = ['-use_fast_math', f'-I{os.path.join(base_path, "krnls")}']
if USE_ACES:
    bloom_opts.append('-DUSE_ACES')
if NOT_USE_S_CURVE:
    bloom_opts.append('-DNOT_USE_S_CURVE')
bloom_module = cp.RawModule(code=bloom_source, options=tuple(bloom_opts))
gaussH = bloom_module.get_function("gaussianBlurH")
gaussW = bloom_module.get_function("gaussianBlurW")
bloom = bloom_module.get_function("compositeBloom")
bright = bloom_module.get_function("extractBright")
downsample2x = bloom_module.get_function("downsample2x")
# debug = bloom_module.get_function("debugOutput")
print('  Kernel 编译完成')

output_dir = os.path.join(base_path, output_dir)
os.makedirs(output_dir, exist_ok=True)
manifest_path = start_manifest(output_dir, 'single_gpu', cfg,
                               {'camera_motion': 'schwarzschild_free_fall',
                                'frame_interval': 'camera_proper_time',
                                'start_frame': start_frame})



tmp_blur_buf = cp.zeros((h // 2, w // 2, 4), dtype=cp.float32)
tmp_blur_tex, tmp_blur_surf = create_texture_surface_union_2d(tmp_blur_buf, 4, (1,1,1,1))




t_val = camera.time



target_size = 8
num_levels = int(np.round(np.log2(min(w, h) / target_size)))
num_levels = max(1, num_levels)
print(f"降采样层数: {num_levels} 层")
blur_scale=1
print(f"模糊因子: {blur_scale:.4f}")
down_texs = []
down_surfs=[]
down_resolutions = []
curr_w, curr_h = w, h
for i in range(num_levels):
    curr_w = max(1, curr_w // 2)
    curr_h = max(1, curr_h // 2)
    buf = cp.zeros((curr_h,curr_w, 4), dtype=cp.float32)
    buf_tex,buf_surf = create_texture_surface_union_2d(buf,4,(1,1,1,1))
    down_resolutions.append((curr_w, curr_h))
    down_texs.append(buf_tex)
    down_surfs.append(buf_surf)
tmp_texs = []
tmp_surfs = []
for i in range(num_levels):
    curr_w, curr_h = down_resolutions[i]
    # 创建一个尺寸与当前级完全契合的临时数组
    tmp_buf = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
    t_tex, t_surf = create_texture_surface_union_2d(tmp_buf, 4, (1, 1, 1, 1))
    tmp_texs.append(t_tex)
    tmp_surfs.append(t_surf)

frame_intermediate_result = cp.empty((h,w, 4), dtype=cp.float32)    # 光追核心的裸hdr数据
frame_inter_tex,frame_inter_surf = create_texture_surface_union_2d(frame_intermediate_result,4,(1,1,0,1))
del frame_intermediate_result
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

bloom_threshold = np.float32(12)

# block_x, block_y 由 cfg/offline.py 提供
grid_x = (w + block_x - 1) // block_x
grid_y = (h + block_y - 1) // block_y
down_grids = [((w + block_x - 1) // block_x, (h + block_y - 1) // block_y) for w, h in down_resolutions]
print(down_resolutions)
print(down_grids)

kernel_args = (
    cp.uint64(frame_inter_surf.ptr),
    cp.uint64(tex_handle.ptr),        # 天空盒
    cp.uint64(tex_prebaked.ptr),      # prebaked_disk
    cp.uint64(tex_handle_color.ptr),  # lut_color

    cp.float32(t_val),                # time
    cp.float32(cam_pos_init[0]), cp.float32(cam_pos_init[1]), cp.float32(cam_pos_init[2]),
    cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
    cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
    cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
    cp.float32(vx), cp.float32(vy), cp.float32(vz),

    cp.int32(w), cp.int32(h),
    cp.float32(w * PIXEL_PITCH), cp.float32(h * PIXEL_PITCH),  # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(step),
    cp.int32(maxstep),
    cp.int32(SSAA_COUNT),                   # jitternum
    cp.int32(1),                            # frames
)

bright_buf = cp.zeros((h,w,4), dtype=cp.float32)
bright_buf_tex,bright_buf_surf = create_texture_surface_union_2d(bright_buf,4,(1,1,0,1))
del bright_buf
# tmp_blur_buf = cp.empty((h * w * 4), dtype=cp.float32)
tex_ptrs = [tex.ptr for tex in down_texs]
tex_group = cp.array(tex_ptrs, dtype=cp.uint64)



print('\n正在录制后处理 CUDA Graph...')
capture_stream = cp.cuda.Stream(non_blocking=True)
with capture_stream:
    capture_stream.begin_capture()  # 开启捕获
    bright(
        (grid_x, grid_y), (block_x, block_y),
        (
            cp.uint64(frame_inter_tex.ptr),
            cp.uint64(bright_buf_surf.ptr),
            cp.int32(w), cp.int32(h),
            cp.float32(1.0), cp.float32(bloom_threshold)
        )
    )
    prev_tex_ptr = bright_buf_tex.ptr
    for i in range(num_levels):
        out_w, out_h = down_resolutions[i]
        grid_out_x = (out_w + block_x - 1) // block_x
        grid_out_y = (out_h + block_y - 1) // block_y
        downsample2x(
            (grid_out_x, grid_out_y), (block_x, block_y),
            (cp.uint64(prev_tex_ptr), cp.uint64(down_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h))
        )
        gaussH(
            (grid_out_x, grid_out_y), (block_x, block_y),
            (cp.uint64(tmp_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h), cp.uint64(tex_ptrs[i]), cp.float32(blur_scale))
        )
        gaussW(
            (grid_out_x, grid_out_y), (block_x, block_y),
            (cp.uint64(down_surfs[i].ptr), cp.int32(out_w), cp.int32(out_h), cp.uint64(tmp_texs[i].ptr), cp.float32(blur_scale))
        )
        prev_tex_ptr = cp.uint64(tex_ptrs[i])
    bloom(
        (grid_x, grid_y), (block_x, block_y),
        (
            current_frame_float,
            cp.int32(w), cp.int32(h),
            cp.uint64(frame_inter_tex.ptr),
            tex_group,
            cp.int32(num_levels),
            cp.int32(w), cp.int32(h)
        )
    )

    postprocess_graph = capture_stream.end_capture()  # 结束捕获
print('  后处理 CUDA Graph 录制完成')




print(f"\n开始离线渲染, 总计 {total_frames} 帧, 起始帧号: {start_frame}, 输出目录: {output_dir}")




   

# Finish initial uploads/clears before the non-blocking render stream reads them.
cp.cuda.get_current_stream().synchronize()
render_stream = cp.cuda.Stream(non_blocking=True)
kernel_args_dynamic = list(kernel_args)
camera_frames = []
frames_rendered = 0
for render_idx in range(1, total_frames + 1):
    frame_idx = start_frame + render_idx - 1
    start_time = time.time()
    cp.cuda.profiler.start()
    with render_stream:
        # 1. 核心光追
        trace_rays_kernel((grid_x, grid_y), (block_x, block_y), tuple(kernel_args_dynamic))
        # 2. 紧接着执行后处理 (GPU 会自动等光追画完再处理 Graph,绝不抢跑)
        postprocess_graph.launch(stream=render_stream)
        # debug((grid_x, grid_y), (block_x, block_y),(current_frame_float,w,h,frame_inter_tex.ptr))

    # ====== 【修复点 3】必须同步!让 CPU 停下来等 GPU 把所有活干完再截图 ======
    cp.cuda.profiler.stop()
    render_stream.synchronize()

    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
    # img_bgr_np[502:612,512:522,0]=255
    # img_bgr_np[602,512,1]=0
    # img_bgr_np[602,512,2]=255
    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    if not cv2.imwrite(output_path, img_bgr_np):
        raise OSError(f'Could not save frame: {output_path}')
    frames_rendered += 1
    camera_frames.append({'frame': frame_idx, 'proper_time': camera.proper_time,
                          'time': camera.time, 'position': camera.position.tolist(),
                          'velocity': camera.velocity.tolist()})

    elapsed = time.time() - start_time


    print(f"[Frame {frame_idx:05d} | {render_idx}/{total_frames}] {elapsed:.3f} s | "
          f"t={camera.time:.4f} tau={camera.proper_time:.4f} "
          f"r={np.linalg.norm(camera.position):.4f} "
          f"beta={np.linalg.norm(camera.velocity):.6f}")
    if render_idx < total_frames and not update_after_frame(camera, kernel_args_dynamic, render_idx, d_tau):
        print('相机已接近视界（r=0.55），停止后续帧渲染。')
        break


    gc.collect()

print("\n渲染完毕!")
trajectory_path = manifest_path.with_name(manifest_path.stem + '_camera.json')
with open(trajectory_path, 'w', encoding='utf-8') as f:
    json.dump(camera_frames, f, ensure_ascii=False, indent=2)
finish_manifest(manifest_path, 'complete', {'frames_rendered': frames_rendered,
                'last_frame': start_frame + frames_rendered - 1 if frames_rendered else None,
                'stopped_near_horizon': camera.stopped, 'camera_trajectory': trajectory_path.name})
