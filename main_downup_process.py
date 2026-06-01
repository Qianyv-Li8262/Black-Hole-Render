import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture, runtime
import os, sys


def create_texture_object(img_cp, num_of_channels, is_half=False):
    """创建 2D 纹理（带自动 padding），用于 2D LUT / 天空盒，支持 FP16 和 FP32"""
    h, w, c = img_cp.shape
    alignment = 256

    # 1. 根据是否为 half (FP16) 决定每个像素的字节数 (4通道) 以及数组数据类型
    bytes_per_pixel = 8 if is_half else 16
    dtype = cp.float16 if is_half else cp.float32

    # 2. 计算满足 256 字节对齐的 pitch 和宽度
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    
    # 3. 创建带有填充的目标数组并复制数据
    rgba = cp.zeros((h, padded_w, 4), dtype=dtype)
    rgba[:, :w, :num_of_channels] = img_cp

    # 4. 根据 is_half 创建对应的通道描述符
    ch_fmt = (
        texture.ChannelFormatDescriptor(16, 16, 16, 16, runtime.cudaChannelFormatKindFloat) 
        if is_half else 
        texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    )

    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D,
        arr=rgba,
        chDesc=ch_fmt,
        width=w,
        height=h,
        pitchInBytes=rgba.strides[0],  # 自动对应 rgba 实际每行的字节数 (等同于 pitch_bytes)
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
    ch_fmt = (
        texture.ChannelFormatDescriptor(16, 16, 16, 16, runtime.cudaChannelFormatKindFloat) 
        if is_half else 
        texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    )
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


def create_3d_texture_from_npy(data_gpu, is_half=False):
    R, Z, PHI, C = data_gpu.shape
    assert C == 4, f"Expected 4-channel data, got {C}"

    # 1. 根据 is_half 参数选择目标数据类型 (cp.float16 或 cp.float32)
    target_dtype = cp.float16 if is_half else cp.float32
    if data_gpu.dtype != target_dtype:
        data_gpu = data_gpu.astype(target_dtype, copy=False)
    
    data_contiguous = cp.ascontiguousarray(data_gpu)

    # 2. 匹配对应的 16-bit 或 32-bit 四通道描述符
    ch_desc = (
        texture.ChannelFormatDescriptor(16, 16, 16, 16, runtime.cudaChannelFormatKindFloat)
        if is_half else
        texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    )

    # 3. 创建 3D CUDA array 并执行数据拷贝
    # (此时如果 is_half=True，每个 3D 像素单元的物理大小会自动变更为 8 字节)
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

    return texture.TextureObject(res_desc, tex_desc)


_tex_pins = []  # 阻止 GC 回收纹理底层的 padded array，每帧 sync 后清空


def make_tex_from_flat(buf_flat, h_img, w_img, is_half=False):
    """从 flat (h*w, 4) buffer 创建 2D 纹理（自动 256B 对齐 padding，兼容任意分辨率）"""
    img = buf_flat.reshape((h_img, w_img, 4))
    tex, rgba = create_texture_object(img, 4, is_half)
    _tex_pins.append(rgba)  # 防止 padded array 被 GC 回收导致纹理悬空
    return tex



base_path = os.path.dirname(os.path.abspath(__file__))


print('正在加载天空盒...')
img_bgr = cv2.imread(os.path.join(base_path, 'starmap_random_2020_16k.exr'),
                     cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    print(f"错误：无法加载天空盒！")
    exit()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) * 1
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


print('正在加载预烘焙吸积盘纹理...')
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise.npy'))
ishalf=True
tex_prebaked = create_3d_texture_from_npy(cp.asarray(prebaked_data, dtype=cp.float16),ishalf)
print(f"  数据 shape: {prebaked_data.shape}  dtype: {'half' if ishalf else 'float32'}")
del prebaked_data
print('  吸积盘 3D 纹理就绪')


print('正在加载颜色 LUT...')
tex_handle_color, _ = create_texture_object(
    cp.asarray(np.load(os.path.join(base_path, 'color_lut2.npy')).astype(cp.float16)), 3,True)
print('  颜色 LUT 就绪')


print('正在编译 CUDA kernel...')
kernel_path = os.path.join(base_path, "blackholekernel3_prebaked.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
module = cp.RawModule(code=cuda_source, options=('-use_fast_math','-lineinfo'))
trace_rays_kernel = module.get_function("blackholekernel")


bloom_path = os.path.join(base_path, "postprocess_gemini.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',))
extract_bright_kernel = bloom_module.get_function("extract_bright_kernel")
bloom_path = os.path.join(base_path, "postprocess_downup.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
downup_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',))
down = downup_module.get_function("tap13_downsample")
up1=downup_module.get_function("tent_upsampling_kernel1")
up2 = downup_module.get_function("tent_upsampling_kernel2")
final = downup_module.get_function("combine_hdr_bloom_kernel")

print('  Kernel 编译完成')


w, h = 8192,4320
total_frames = 1
start_t = 50
SSAA_COUNT = 16

output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)

cam_pos_init = np.array([  3.4581583 ,-22.43106    , 3.8267765], dtype=np.float32)
r0 = np.linalg.norm(cam_pos_init)
dir_unit = cam_pos_init / r0

r = r0
t_val = start_t
tau = 0.0
d_tau = 0.1

cam_yaw, cam_pitch, cam_roll = -5.02, -0.28, 0
focal_length = 1.1716

vx,vy,vz=0,0,0
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



frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

# ---- 降采样金字塔分辨率 ----
w1, h1 = w // 2, h // 2      # 4096, 2160
w2, h2 = w1 // 2, h1 // 2    # 2048, 1080
w3, h3 = w2 // 2, h2 // 2    # 1024, 540
w4, h4 = w3 // 2, h3 // 2    # 512,  270
w5, h5 = w4 // 2, h4 // 2    # 256,  135

# ---- 降采样缓冲区 ----
u1_buf = cp.empty((h1 * w1 * 4), dtype=cp.float32)
u2_buf = cp.empty((h2 * w2 * 4), dtype=cp.float32)
u3_buf = cp.empty((h3 * w3 * 4), dtype=cp.float32)
u4_buf = cp.empty((h4 * w4 * 4), dtype=cp.float32)
u5_buf = cp.empty((h5 * w5 * 4), dtype=cp.float32)

# ---- 升采样临时缓冲区 (最大尺寸 = u1 级别) ----
upsample_temp = cp.empty((h1 * w1 * 4), dtype=cp.float32)
upsample_out  = cp.empty((h1 * w1 * 4), dtype=cp.float32)

bloom_threshold = np.float32(5.0)
scatter = np.float32(0.5)
bloom_intensity = np.float32(1.0)

block_x, block_y = 32, 8
grid_x = (w + block_x - 1) // block_x
grid_y = (h + block_y - 1) // block_y


kernel_args = (
    frame_intermediate_result,
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
    cp.float32(4.096), cp.float32(2.160),        # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(0.1),                        # step
    cp.int32(2000),                         # maxstep
    cp.int32(SSAA_COUNT),                   # jitternum
    cp.int32(1),                            # frames
)



print(f"\n开始离线渲染, 总计 {total_frames} 帧, 输出目录: {output_dir}")

for frame_idx in range(1, total_frames + 1):
    cam_pos = r * dir_unit
    start_time = time.time()


    kernel_args_dynamic = list(kernel_args)
    kernel_args_dynamic[4] = cp.float32(t_val)  # time
    kernel_args_dynamic[5] = cp.float32(cam_pos[0])
    kernel_args_dynamic[6] = cp.float32(cam_pos[1])
    kernel_args_dynamic[7] = cp.float32(cam_pos[2])
    kernel_args_dynamic[17] = cp.float32(vx)
    kernel_args_dynamic[18] = cp.float32(vy)
    kernel_args_dynamic[19] = cp.float32(vz)
    kernel_args_dynamic[-1] = cp.int32(frame_idx)  # frames
    cp.cuda.profiler.start()
    trace_rays_kernel((grid_x, grid_y), (block_x, block_y), tuple(kernel_args_dynamic))
    cp.cuda.profiler.stop()
    extract_bright_kernel((grid_x, grid_y), (block_x, block_y),
        (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h),
         np.float32(1), bloom_threshold))

    # =====================================================================
    # 降采样金字塔: bright_buf → u1 → u2 → u3 → u4 → u5
    # =====================================================================
    # Level 0→1: 8192x4320 → 4096x2160
    tex_src = make_tex_from_flat(bright_buf, h, w)
    down((grid_x, grid_y), (block_x, block_y),
         (tex_src, u1_buf, np.int32(w), np.int32(h), np.int32(w1), np.int32(h1)))

    # Level 1→2: 4096x2160 → 2048x1080
    gx1 = (w1 + 31) // 32; gy1 = (h1 + 7) // 8
    tex_src = make_tex_from_flat(u1_buf, h1, w1)
    down((gx1, gy1), (block_x, block_y),
         (tex_src, u2_buf, np.int32(w1), np.int32(h1), np.int32(w2), np.int32(h2)))

    # Level 2→3: 2048x1080 → 1024x540
    gx2 = (w2 + 31) // 32; gy2 = (h2 + 7) // 8
    tex_src = make_tex_from_flat(u2_buf, h2, w2)
    down((gx2, gy2), (block_x, block_y),
         (tex_src, u3_buf, np.int32(w2), np.int32(h2), np.int32(w3), np.int32(h3)))

    # Level 3→4: 1024x540 → 512x270
    gx3 = (w3 + 31) // 32; gy3 = (h3 + 7) // 8
    tex_src = make_tex_from_flat(u3_buf, h3, w3)
    down((gx3, gy3), (block_x, block_y),
         (tex_src, u4_buf, np.int32(w3), np.int32(h3), np.int32(w4), np.int32(h4)))

    # Level 4→5: 512x270 → 256x135
    gx4 = (w4 + 31) // 32; gy4 = (h4 + 7) // 8
    tex_src = make_tex_from_flat(u4_buf, h4, w4)
    down((gx4, gy4), (block_x, block_y),
         (tex_src, u5_buf, np.int32(w4), np.int32(h4), np.int32(w5), np.int32(h5)))

    # =====================================================================
    # 升采样 + 混合金字塔: u5 → ... → u1
    # =====================================================================
    downsample_bufs = [u4_buf, u3_buf, u2_buf, u1_buf]
    level_dims = [(w5, h5), (w4, h4), (w3, h3), (w2, h2), (w1, h1)]
    # 升采样目标分辨率列表（从 u4 到 u1）
    target_dims = [(w4, h4), (w3, h3), (w2, h2), (w1, h1)]

    current_flat = u5_buf  # 起点: u5 (256x135)

    for idx, (tw, th) in enumerate(target_dims):
        cur_h, cur_w = level_dims[idx][1], level_dims[idx][0]  # 当前小图尺寸

        # 创建当前小图的纹理
        tex_small = make_tex_from_flat(current_flat, cur_h, cur_w)

        # up1: 纹理 → temp (双线性上采样到 2x 分辨率)
        gx_up1 = (tw + 31) // 32; gy_up1 = (th + 7) // 8
        up1((gx_up1, gy_up1), (block_x, block_y),
            (tex_small, upsample_temp, np.int32(cur_w), np.int32(cur_h),
             np.int32(tw), np.int32(th)))

        # up2: 3x3 tent filter
        gx_up2 = (tw + 31) // 32; gy_up2 = (th + 31) // 32
        up2((gx_up2, gy_up2), (32, 32, 1),
            (upsample_temp, upsample_out, np.int32(tw), np.int32(th)))

        # 混合: upsample_out * scatter + 当前级别的降采样图 (cupy)
        n_pix = th * tw
        up_view = upsample_out[:n_pix * 4].reshape((th, tw, 4))
        dn_view = downsample_bufs[idx][:n_pix * 4].reshape((th, tw, 4))
        blended = up_view * scatter + dn_view
        current_flat = blended.reshape((-1,))

    # =====================================================================
    # 最终合成: u1 级别的 bloom 纹理 + 原始 HDR → 色调映射输出
    # =====================================================================
    tex_bloom_final = make_tex_from_flat(current_flat, h1, w1)
    final((grid_x, grid_y), (block_x, block_y),
          (frame_intermediate_result, tex_bloom_final, current_frame_float,
           np.int32(w), np.int32(h), bloom_intensity))

    cp.cuda.Device().synchronize()
    _tex_pins.clear()  # 所有 kernel 已完成，释放 padded array 引用

    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)

    output_path = os.path.join(output_dir, f"downup.png")
    cv2.imwrite(output_path, img_bgr_np)



    elapsed = time.time() - start_time


    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] {elapsed:.3f} s | "
          f"tau={tau:.1f} r={r:.4f} t={t_val:.2f} beta={np.sqrt(vx**2+vy**2+vz**2):.4f}")

print("\n渲染完毕！")
