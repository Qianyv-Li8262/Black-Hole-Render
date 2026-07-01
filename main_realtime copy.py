'''
This script is only for finding a good position to make pictures.
I don't know the way to do temporal antialiasing(I'm a noob!)when the disk is moving so I give up real-time rendering and turned to offline rendering.
'''

import numpy as np
import cupy as cp
import cv2
import time
from cuda_tex import *
import glfw
from cupyx.scipy.ndimage import gaussian_filter
import os, sys
from zero_copy_window import ZeroCopyWindow

base_path = os.path.dirname(os.path.abspath(__file__))
img_file_path = os.path.join(base_path, 'starmap_random_2020_16k.exr') 
img_bgr = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)   

if img_bgr is None:
    print(f"错误:无法在路径 {img_file_path} 找到背景图片!")
    exit() 

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)*100
img_float = cp.asarray(img_rgb.astype(np.float32))
del img_rgb
img_rgba = cp.zeros((*img_float.shape[:2], 4), dtype=cp.float16)
img_rgba[:, :, :3] = img_float
tex_handle = create_texture_array_2d(img_rgba, 4, (1, 1, 1, 1), True)

print('正在加载预烘焙吸积盘纹理...')
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise_npgs.npy'))
ishalf = True
tex_prebaked = create_texture_array_3d(cp.asarray(prebaked_data, dtype=cp.float16), 4, (1, 1, 1, 1, 1), is_half=ishalf)
del prebaked_data

colorlut_file_path = os.path.join(base_path, 'color_lut2.npy')
lut_color = cp.load(colorlut_file_path).astype(cp.float16)

lut_rgba = cp.zeros((*lut_color.shape[:2], 4), dtype=cp.float16)
lut_rgba[:, :, :3] = lut_color
tex_handle_color = create_texture_array_2d(lut_rgba, 4, (1, 1, 1, 1), True)

print('正在编译 CUDA kernel...')
kernel_path = os.path.join(base_path, "blackholekernel3_prebaked copy.cu")
# kernel_path = os.path.join(base_path, "glm_wish_coding_try1.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

include_dir = os.path.dirname(os.path.abspath(kernel_path)) 
module = cp.RawModule(code=cuda_source, options=(
    '-use_fast_math',
    '-lineinfo',
    '-DNO_DEPTH_JITTER',

    '-I', include_dir
))
trace_rays_kernel = module.get_function("blackholekernel")

bloom_path = os.path.join(base_path, "postprocess_downup copy.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',))
gaussH = bloom_module.get_function("gaussianBlurH")
gaussW = bloom_module.get_function("gaussianBlurW")
bloom = bloom_module.get_function("compositeBloom")
bright = bloom_module.get_function("extractBright")
downsample2x = bloom_module.get_function("downsample2x")

# 超参数与窗口初始化
w, h = 4096,2160

cam_pos = np.array([38.71, -44.96, 5.18], dtype=np.float32)


cam_yaw, cam_pitch, cam_roll = -4.04, -0.09, 0
focal_length = 3

move_speed = 0.05
turn_speed = 0.01
focus_speed = 1.02
jitnum = 1

window = ZeroCopyWindow(w, h, 'Real-time Viewfinder')

# ==================== [静态缓冲区配置与 Union 纹理创建] ====================
# 1. 核心光追输出缓冲区 (Surface)
frame_intermediate_result = cp.empty((h, w, 4), dtype=cp.float32)
frame_inter_tex, frame_inter_surf = create_texture_surface_union_2d(frame_intermediate_result, 4, (1, 1, 0, 1))

# 2. 高亮区域缓冲区
bright_buf = cp.zeros((h, w, 4), dtype=cp.float32)
bright_buf_tex, bright_buf_surf = create_texture_surface_union_2d(bright_buf, 4, (1, 1, 0, 1))

target_size = 8
num_levels = int(np.round(np.log2(min(w, h) / target_size)))
num_levels = max(1, num_levels)
blur_scale = (np.float32(w / 4096.0) - 1) * 1.5 + 1

down_texs = []
down_surfs = []
down_resolutions = []
curr_w, curr_h = w, h
for i in range(num_levels):
    curr_w = max(1, curr_w // 2)
    curr_h = max(1, curr_h // 2)
    buf = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
    buf_tex, buf_surf = create_texture_surface_union_2d(buf, 4, (1, 1, 1, 1))
    down_resolutions.append((curr_w, curr_h))
    down_texs.append(buf_tex)
    down_surfs.append(buf_surf)

tmp_texs = []
tmp_surfs = []
for i in range(num_levels):
    curr_w, curr_h = down_resolutions[i]
    tmp_buf = cp.zeros((curr_h, curr_w, 4), dtype=cp.float32)
    t_tex, t_surf = create_texture_surface_union_2d(tmp_buf, 4, (1, 1, 1, 1))
    tmp_texs.append(t_tex)
    tmp_surfs.append(t_surf)

tex_ptrs = [tex.ptr for tex in down_texs]
tex_group = cp.array(tex_ptrs, dtype=cp.uint64)

bloom_threshold = np.float32(1.7)
bloom_radius = np.int32(20)
bloom_sigma = np.float32(8.0)
bloom_strength = np.float32(1.5)

block_x, block_y = 32, 8
grid_x = (w + block_x - 1) // block_x
grid_y = (h + block_y - 1) // block_y

# 【最终输出静态线性缓冲】：1D uint8 数组，与 PBO 完全契合
static_output_buf = cp.empty((h * w * 4), dtype=cp.uint8)

# ==================== [录制直通后处理 CUDA Graph] ====================
print('\n正在录制后处理 CUDA Graph...')
capture_stream = cp.cuda.Stream(non_blocking=True)
with capture_stream:
    capture_stream.begin_capture()

    # 【重要改动】：直接把光追生成的纹理 frame_inter_tex 喂给提取器，摒弃断层的累加器
    bright(
        (grid_x, grid_y), (block_x, block_y),
        (
            cp.uint64(frame_inter_tex.ptr),
            cp.uint64(bright_buf_surf.ptr),
            cp.int32(w), cp.int32(h),
            cp.float32(1.0),
            cp.float32(bloom_threshold)
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

    # 混合原图 (frame_inter_tex) 与 Bloom，并输出到线性静态缓冲
    bloom(
        (grid_x, grid_y), (block_x, block_y),
        (
            static_output_buf,
            cp.int32(w), cp.int32(h),
            cp.uint64(frame_inter_tex.ptr),
            tex_group,
            cp.int32(num_levels),
            cp.int32(w), cp.int32(h)
        )
    )

    postprocess_graph = capture_stream.end_capture()
print('  后处理 CUDA Graph 录制完成')
# ======================================================================

tot_pixels = w * h
frames = 1
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

def update_camera_vectors(yaw, pitch, roll):
    fwd_x = np.cos(yaw) * np.cos(pitch)
    fwd_y = np.sin(yaw) * np.cos(pitch)
    fwd_z = np.sin(pitch)
    fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
    fwd /= np.linalg.norm(fwd)
    
    right0 = np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right0)
    right0 = right0 / right_norm if right_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    up0 = np.cross(right0, fwd)
    up0 /= np.linalg.norm(up0)
    
    right = right0 * np.cos(roll) + up0 * np.sin(roll)
    up = up0 * np.cos(roll) - right0 * np.sin(roll)
    return fwd, right, up

fwd, right, up = update_camera_vectors(cam_yaw, cam_pitch, cam_roll)

t = 15.0

# 创建高效异步的主渲染流
render_stream = cp.cuda.Stream(non_blocking=True)

while not window.should_close():
    camera_moved = False
    
    if glfw.KEY_W in window.key_pressed: cam_pos += fwd * move_speed; camera_moved = True
    if glfw.KEY_S in window.key_pressed: cam_pos -= fwd * move_speed; camera_moved = True
    if glfw.KEY_D in window.key_pressed: cam_pos += right * move_speed; camera_moved = True
    if glfw.KEY_A in window.key_pressed: cam_pos -= right * move_speed; camera_moved = True
    if glfw.KEY_UP in window.key_pressed: cam_pos += up * move_speed; camera_moved = True
    if glfw.KEY_DOWN in window.key_pressed: cam_pos -= up * move_speed; camera_moved = True
    if glfw.KEY_E in window.key_pressed: cam_yaw -= turn_speed; camera_moved = True
    if glfw.KEY_Q in window.key_pressed: cam_yaw += turn_speed; camera_moved = True
    if glfw.KEY_R in window.key_pressed: cam_pitch += turn_speed; camera_moved = True
    if glfw.KEY_F in window.key_pressed: cam_pitch -= turn_speed; camera_moved = True
    if glfw.KEY_Z in window.key_pressed: cam_roll -= turn_speed; camera_moved = True
    if glfw.KEY_C in window.key_pressed: cam_roll += turn_speed; camera_moved = True
    if glfw.KEY_G in window.key_pressed: focal_length /= focus_speed; camera_moved = True
    if glfw.KEY_T in window.key_pressed: focal_length *= focus_speed; camera_moved = True

    if camera_moved:
        frames = 1
        cam_pitch = np.clip(cam_pitch, -np.pi/2 + 0.001, np.pi/2 - 0.001)
        fwd, right, up = update_camera_vectors(cam_yaw, cam_pitch, cam_roll)

    # 1. 在流内全速启动渲染流水线
    with render_stream:
        # 光追写入 Surface
        trace_rays_kernel(
            (grid_x, grid_y), (block_x, block_y), 
            (
                cp.uint64(frame_inter_surf.ptr), cp.uint64(tex_handle.ptr), cp.uint64(tex_prebaked.ptr), cp.uint64(tex_handle_color.ptr), cp.float32(t),
                cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
                cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
                cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
                cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
                cp.float32(0), cp.float32(0), cp.float32(0),
                cp.int32(w), cp.int32(h),
                cp.float32(3.2), cp.float32(2.0), cp.float32(focal_length), cp.float32(0.1), cp.int32(2000), cp.int32(jitnum), cp.int32(frames)
            )
        )
        # Graph读取 Surface，处理Bloom，输出给线性 uint8 静态数组
        postprocess_graph.launch(stream=render_stream)

    # 2. 必须同步！等待流水线走完，此时 static_output_buf 画面确立
    render_stream.synchronize()
    
    # 3. 映射 PBO，进行内存级极速复刻
    current_frame_pbo = window.map_pbo()
    
    # PBO与输出都是扁平的 1D uint8数组，直接 1:1 零开销倾倒过去
    current_frame_pbo[:] = static_output_buf
    
    # 4. 解除映射并由 OpenGL 画在屏幕上
    window.unmap_and_draw()

    t += 0.5
    frames += 1
    
    if frames % 10 == 0 or camera_moved:
        print(f"CamPos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}] | Pitch: {cam_pitch:.3f} | Yaw: {cam_yaw:.3f} | Focus: {focal_length:.3f}")

window.destroy()
print('Done.')