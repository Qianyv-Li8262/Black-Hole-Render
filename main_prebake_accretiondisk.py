"""
黑洞光线追踪 — 预烘焙噪声版本
基于 gmn try3.py，使用 blackholekernel3_prebaked.cu + 3D 预烘焙吸积盘纹理
"""
import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture, runtime
import os, sys

# ============================================================
# 纹理工具函数
# ============================================================

import cupy as cp
from cupy.cuda import runtime, texture

# ============================================================
# 纹理工具函数 (生产级优化版)
# ============================================================

def create_texture_object(img_cp, num_of_channels):
    """创建 2D 纹理（带自动 padding），用于 2D LUT / 天空盒"""
    h, w, c = img_cp.shape
    alignment = 256
    # 每个像素 float32 * 4 通道 = 16 字节
    pitch_bytes = ((w * 16 + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // 16
    
    # 物理分配
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :num_of_channels] = img_cp

    ch_fmt = texture.ChannelFormatDescriptor(
        32, 32, 32, 32,
        runtime.cudaChannelFormatKindFloat
    )
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D,
        arr=rgba,
        chDesc=ch_fmt,
        width=w,
        height=h,
        pitchInBytes=rgba.strides[0],  # 直接利用连续数组的第一维步长作为 Pitch，安全且精准
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1,
    )
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    return tex_obj, rgba


def create_texture_object_nopadding(img_cp_padded, num_of_channels,
                                     h_real, w_real, pitch_bytes=None, is_half=False):
    """创建 2D 纹理（预 padded 数据，用于 half-float 天空盒）"""
    ch_fmt = (
        texture.ChannelFormatDescriptor(16, 16, 16, 16, runtime.cudaChannelFormatKindFloat) 
        if is_half else 
        texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    )
    
    # 若未传 pitch_bytes，则自动提取数组的第一维步长作为 Pitch 字节数
    if pitch_bytes is None:
        pitch_bytes = img_cp_padded.strides[0]

    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D,
        arr=img_cp_padded,
        chDesc=ch_fmt,
        width=w_real,
        height=h_real,
        pitchInBytes=pitch_bytes,
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1,
    )
    return texture.TextureObject(res_desc, tex_desc)


def create_3d_texture_from_npy(data_gpu):
    """
    将 CuPy 4D 数组 (R, Z, PHI, 4) 创建为 CUDA 3D 纹理。
    纹理 extent = (width=PHI, height=Z, depth=R)
    tex3D 坐标: (x→phi, y→z, z→r_disk)
    """
    R, Z, PHI, C = data_gpu.shape
    assert C == 4, f"Expected 4-channel data, got {C}"

    if data_gpu.dtype != cp.float32:
        data_gpu = data_gpu.astype(cp.float32, copy=False)
    data_contiguous = cp.ascontiguousarray(data_gpu)

    ch_desc = texture.ChannelFormatDescriptor(
        32, 32, 32, 32,
        runtime.cudaChannelFormatKindFloat
    )

    # 现代 CuPy 稳定版本直接实例化 CUDAarray
    cuda_arr = texture.CUDAarray(ch_desc, PHI, Z, R)
    data_for_copy = data_contiguous.reshape(R, Z, PHI * C)
    cuda_arr.copy_from(data_for_copy)

    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=cuda_arr
    )
    tex_desc = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp,) * 3,
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1,
    )
    
    # 直接返回底层的 TextureObject 实例
    return texture.TextureObject(res_desc, tex_desc)

# ============================================================
# 广义相对论测地线解析物理求解器（摆线反解法）
# ============================================================
def update_camera_physics_analytical(tau, r_start, dir_unit, fwd, right, up, d_tau=1.0):
    M = 1.0
    R_start = r_start + 1.0 + 0.25 / r_start
    tau_max = (np.sqrt(2.0) / 3.0) * (R_start**1.5)

    if tau >= tau_max:
        r_next = 0.51
        beta_mag = 0.999
    else:
        R = (R_start**1.5 - (1.5 * np.sqrt(2.0)) * tau) ** (2.0 / 3.0)
        R = max(2.0001, R)
        r_next = 0.5 * ((R - 1.0) + np.sqrt((R - 1.0)**2 - 1.0))
        u_next = 1.0 / (2.0 * r_next)
        beta_mag = np.sqrt(2.0 / r_next) / (1.0 + u_next)
        beta_mag = min(0.999, beta_mag)

    beta_global = -beta_mag * dir_unit
    vx_next = np.dot(beta_global, fwd)
    vy_next = np.dot(beta_global, right)
    vz_next = np.dot(beta_global, up)

    u_next = 1.0 / (2.0 * r_next)
    factor = (1.0 + u_next) / (1.0 - u_next)
    gamma = 1.0 / np.sqrt(1.0 - beta_mag**2)
    dt = factor * gamma * d_tau
    return r_next, vx_next, vy_next, vz_next, dt


# ============================================================
# 主程序
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))

# ---- 1. 天空盒纹理 ----
print('正在加载天空盒...')
img_bgr = cv2.imread(os.path.join(base_path, 'starmap_random_2020_16k.exr'),
                     cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    print(f"错误：无法加载天空盒！")
    exit()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) * 300
del img_bgr
img_float = img_rgb.astype(np.float16)
del img_rgb
hh, ww, cc = img_float.shape
pitch_bytes = ((ww * 8 + 255) // 256) * 256
padded_w = pitch_bytes // 8
rgba = np.zeros((hh, padded_w, 4), dtype=np.float16)
rgba[:, :ww, :3] = img_float
tex_handle = create_texture_object_nopadding(cp.array(rgba), 3, hh, ww, pitch_bytes, True)
del img_float, rgba
print('  天空盒纹理就绪')

# ---- 2. 预烘焙吸积盘 3D 纹理（替代原来的 lut_physics） ----
print('正在加载预烘焙吸积盘纹理...')
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise.npy'))
print(f'  数据 shape: {prebaked_data.shape}  dtype: {prebaked_data.dtype}')
tex_prebaked = create_3d_texture_from_npy(cp.asarray(prebaked_data, dtype=cp.float32))
del prebaked_data
print('  吸积盘 3D 纹理就绪')

# ---- 3. 颜色 LUT ----
print('正在加载颜色 LUT...')
tex_handle_color, _ = create_texture_object(
    cp.asarray(np.load(os.path.join(base_path, 'color_lut2.npy')).astype(cp.float32)), 3)
print('  颜色 LUT 就绪')

# ---- 4. 编译 CUDA kernel ----
print('正在编译 CUDA kernel...')
kernel_path = os.path.join(base_path, "blackholekernel3_prebaked.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
trace_rays_kernel = module.get_function("blackholekernel")

# Bloom 后处理（复用原文件）
bloom_path = os.path.join(base_path, "postprocess_gemini.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',))
extract_bright_kernel = bloom_module.get_function("extract_bright_kernel")
blur_x_kernel = bloom_module.get_function("blur_x_kernel")
blur_y_fuse_kernel = bloom_module.get_function("blur_y_fuse_postprocess_kernel")
print('  Kernel 编译完成')

# ---- 5. 渲染参数 ----
w, h = 3200, 2000
total_frames = 1
start_t = 15
SSAA_COUNT = 64

output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)

cam_pos_init = np.array([10.0, 0.0, 0.0], dtype=np.float32)
r0 = np.linalg.norm(cam_pos_init)
dir_unit = cam_pos_init / r0

r = r0
t_val = start_t
tau = 0.0
d_tau = 0.1

cam_yaw, cam_pitch, cam_roll = -3.14, -0.1488899, 0.333
focal_length = 1

# 提前计算 fwd/right/up，供物理求解器初始化速度
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
fwd_z = np.sin(cam_pitch)
fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
fwd /= np.linalg.norm(fwd)
right0 = np.cross(fwd, world_up)
right0 /= max(np.linalg.norm(right0), 1e-6)
up0 = np.cross(right0, fwd)
up0 /= np.linalg.norm(up0)
right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)

# 从物理求解器获取初始速度（tau=0），确保与后续帧速度连续
_, vx, vy, vz, _ = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)

# 缓存
frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
blur_x_tmp = cp.empty((h * w * 4), dtype=cp.float32)
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

bloom_threshold = np.float32(1.7)
bloom_radius = np.int32(20)
bloom_sigma = np.float32(8.0)
bloom_strength = np.float32(1.5)

block_x, block_y = 32, 8
grid_x = (w + block_x - 1) // block_x
grid_y = (h + block_y - 1) // block_y

# ---- 6. kernel 参数封装（与原版相比：tex_handle_lut → tex_prebaked） ----
kernel_args = (
    frame_intermediate_result,
    cp.uint64(tex_handle.ptr),        # tex_obj — 天空盒
    cp.uint64(tex_prebaked.ptr),      # prebaked_disk — 3D 吸积盘纹理（替代 lut_physics）
    cp.uint64(tex_handle_color.ptr),  # lut_color

    cp.float32(t_val),                # time
    cp.float32(cam_pos_init[0]), cp.float32(cam_pos_init[1]), cp.float32(cam_pos_init[2]),
    cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
    cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
    cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
    cp.float32(vx), cp.float32(vy), cp.float32(vz),

    cp.int32(w), cp.int32(h),
    cp.float32(3.2), cp.float32(2),        # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(0.1),                        # step
    cp.int32(2000),                         # maxstep
    cp.int32(SSAA_COUNT),                   # jitternum
    cp.int32(1),                            # frames
)

# ---- 7. 预渲染（warm-up） ----
print("\n预渲染（warm-up）...")
start_t=time.time()
cam_pos = r * dir_unit
trace_rays_kernel((grid_x, grid_y), (block_x, block_y), (
    frame_intermediate_result,
    cp.uint64(tex_handle.ptr),        # tex_obj — 天空盒
    cp.uint64(tex_prebaked.ptr),      # prebaked_disk — 3D 吸积盘纹理（替代 lut_physics）
    cp.uint64(tex_handle_color.ptr),  # lut_color

    cp.float32(t_val),                # time
    cp.float32(cam_pos_init[0]), cp.float32(cam_pos_init[1]), cp.float32(cam_pos_init[2]),
    cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
    cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
    cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
    cp.float32(vx), cp.float32(vy), cp.float32(vz),

    cp.int32(w), cp.int32(h),
    cp.float32(3.2), cp.float32(2),        # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(0.1),                        # step
    cp.int32(2000),                         # maxstep
    cp.int32(50),                   # jitternum
    cp.int32(1),                            # frames
))
cp.cuda.Device().synchronize()

extract_bright_kernel((grid_x, grid_y), (block_x, block_y),
    (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h),
     np.float32(1), bloom_threshold))
blur_x_kernel((grid_x, grid_y), (block_x, block_y),
    (bright_buf, blur_x_tmp, np.int32(w), np.int32(h), bloom_radius, bloom_sigma))
blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
    (frame_intermediate_result, blur_x_tmp, current_frame_float, np.int32(w), np.int32(h),
     bloom_radius, bloom_sigma, np.float32(1), bloom_strength))
cp.cuda.Device().synchronize()
img_rgba_cp = current_frame_float.reshape((h, w, 4))
img_rgba_np = cp.asnumpy(img_rgba_cp)
img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
output_path = os.path.join(output_dir, f"pre_render.png")
cv2.imwrite(output_path, img_bgr_np)
end_t=time.time()
warmup_elapsed = (end_t-start_t)/50*SSAA_COUNT
# print(warmup_elapsed)
print(f'预渲染完成，预计每帧用时 {warmup_elapsed:.3f} s' if not warmup_elapsed == 0 else f'预渲染完成')

# ---- 8. 正式渲染循环 ----
print(f"\n开始离线渲染, 总计 {total_frames} 帧, 输出目录: {output_dir}")

for frame_idx in range(1, total_frames + 1):
    cam_pos = r * dir_unit
    start_time = time.time()

    # 更新 kernel 参数中的动态量
    kernel_args_dynamic = list(kernel_args)
    kernel_args_dynamic[4] = cp.float32(t_val)  # time
    kernel_args_dynamic[5] = cp.float32(cam_pos[0])
    kernel_args_dynamic[6] = cp.float32(cam_pos[1])
    kernel_args_dynamic[7] = cp.float32(cam_pos[2])
    kernel_args_dynamic[17] = cp.float32(vx)
    kernel_args_dynamic[18] = cp.float32(vy)
    kernel_args_dynamic[19] = cp.float32(vz)
    kernel_args_dynamic[-1] = cp.int32(frame_idx)  # frames

    trace_rays_kernel((grid_x, grid_y), (block_x, block_y), tuple(kernel_args_dynamic))

    extract_bright_kernel((grid_x, grid_y), (block_x, block_y),
        (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h),
         np.float32(1), bloom_threshold))
    blur_x_kernel((grid_x, grid_y), (block_x, block_y),
        (bright_buf, blur_x_tmp, np.int32(w), np.int32(h), bloom_radius, bloom_sigma))
    blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
        (frame_intermediate_result, blur_x_tmp, current_frame_float, np.int32(w), np.int32(h),
         bloom_radius, bloom_sigma, np.float32(1), bloom_strength))

    cp.cuda.Device().synchronize()

    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)

    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(output_path, img_bgr_np)

    elapsed = time.time() - start_time

    # 更新物理状态
    if frame_idx < total_frames:
        tau += d_tau
        r, vx, vy, vz, dt = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)
        t_val += dt

    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] {elapsed:.3f} s | "
          f"tau={tau:.1f} r={r:.4f} t={t_val:.2f} beta={np.sqrt(vx**2+vy**2+vz**2):.4f}")

print("\n渲染完毕！")
