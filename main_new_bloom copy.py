import numpy as np
import cupy as cp
import cv2
import time
from cuda_tex import *
import os, sys, gc

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


base_path = os.path.dirname(os.path.abspath(__file__))


print('正在加载天空盒...')
# img_bgr = cv2.imread(os.path.join(base_path, 'starmap_random_2020_16k.exr'),
                    #  cv2.IMREAD_UNCHANGED)
img_bgr = cv2.imread(os.path.join(base_path, 'black.bmp'),
                     cv2.IMREAD_UNCHANGED)
if img_bgr is None:
    print(f"错误：无法加载天空盒！")
    exit()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)*100
del img_bgr
img_float = cp.asarray(img_rgb.astype(np.float16))
del img_rgb
tex_handle = create_texture_array_2d(img_float,4,(1,1,1,1),True)
del img_float
print('  天空盒纹理就绪')


print('正在加载预烘焙吸积盘纹理...')
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise_npgs.npy'))
ishalf=True
tex_prebaked = create_texture_array_3d(cp.asarray(prebaked_data, dtype=cp.float16),4,(0,0,1,1,1),ishalf)
print(f"  数据 shape: {prebaked_data.shape}  dtype: {'half' if ishalf else 'float32'}")
del prebaked_data
print('  吸积盘 3D 纹理就绪')


print('正在加载颜色 LUT...')
lut_rgb = np.load(os.path.join(base_path, 'color_lut2.npy'))
lut_rgba = np.ones((*lut_rgb.shape[:-1], 4), dtype=lut_rgb.dtype)
lut_rgba[..., :3] = lut_rgb
img_float = cp.asarray(lut_rgba, dtype=cp.float16)
tex_handle_color = create_texture_array_2d(img_float, 4, (1, 1, 1, 1), True)
print('  颜色 LUT 就绪')


print('正在编译 CUDA kernel...')
kernel_path = os.path.join(base_path, "blackholekernel3_prebaked copy.cu")
# kernel_path = os.path.join(base_path, "glm_wish_coding_try1.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
# 条件编译打开rk4可以显著提升渲染速度（4-5倍，rk4步长大），但是吸积盘的艺术风格会和rk2有差别，而且会存在较严重的采样伪影
# 建议若不是为了速度，应关闭rk4
module = cp.RawModule(code=cuda_source, options=('-use_fast_math',f'-I{base_path}','-lineinfo','-DNO_DEPTH_JITTER','-DRAND_SAMP_DISK'))
trace_rays_kernel = module.get_function("blackholekernel")


bloom_path = os.path.join(base_path, "postprocess_downup copy.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',f'-I{base_path}',))
gaussH = bloom_module.get_function("gaussianBlurH")
gaussW = bloom_module.get_function("gaussianBlurW")
bloom = bloom_module.get_function("compositeBloom")
bright = bloom_module.get_function("extractBright")
downsample2x = bloom_module.get_function("downsample2x")
debug = bloom_module.get_function("debugOutput")
print('  Kernel 编译完成')

#-2.47,-4.47,-2.44     -2.61,-4.84,-0.70
w, h = 3200,2000
total_frames = 1
start_t = 20.5
SSAA_COUNT = 5

output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)



tmp_blur_buf = cp.zeros((h // 2, w // 2, 4), dtype=cp.float32)
tmp_blur_tex, tmp_blur_surf = create_texture_surface_union_2d(tmp_blur_buf, 4, (1,1,1,1))




# cam_pos_init = np.array([-4.824605,  -9.414302 ,  4.8539257], dtype=np.float32)
# r0 = np.linalg.norm(cam_pos_init)
# dir_unit = cam_pos_init / r0

# r = r0
# t_val = start_t
# tau = 0.0
# d_tau = 0.1

# cam_yaw, cam_pitch, cam_roll = -5.24, -0.05, 0
# focal_length = 96


cam_pos_init = np.array([12,-18, 2.4], dtype=np.float32)
r0 = np.linalg.norm(cam_pos_init)
dir_unit = cam_pos_init / r0

r = r0
t_val = start_t
tau = 0.0
d_tau = 0.1

cam_yaw, cam_pitch, cam_roll = -4.24, -0.1, -0.4
focal_length = 4




vx,vy,vz=0,0,0
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
fwd_z = np.sin(cam_pitch)
fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
fwd /= np.linalg.norm(fwd)
right0 = np.cross(fwd, world_up)
right0 /= np.linalg.norm(right0)
up0 = np.cross(right0, fwd)
up0 /= np.linalg.norm(up0)
right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)

target_size = 8
num_levels = int(np.round(np.log2(min(w, h) / target_size)))
num_levels = max(1, num_levels)
print(f"降采样层数: {num_levels} 层")
blur_scale = (np.float32(w / 4096.0)-1)*1.5+1
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
# _, vx, vy, vz, _ = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)
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

bloom_threshold = np.float32(1.7)

block_x, block_y = 32, 8
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
    cp.float32(6.4), cp.float32(4),        # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(0.02),                        # step
    cp.int32(4000),                         # maxstep
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




print(f"\n开始离线渲染, 总计 {total_frames} 帧, 输出目录: {output_dir}")




   

render_stream = cp.cuda.Stream(non_blocking=True)
for frame_idx in range(1, total_frames + 1):
    cam_pos = r * dir_unit
    start_time = time.time()
    # temp_textures=[]
    
    kernel_args_dynamic = list(kernel_args)
    kernel_args_dynamic[4] = cp.float32(t_val)  # time
    kernel_args_dynamic[5] = cp.float32(cam_pos[0])
    kernel_args_dynamic[6] = cp.float32(cam_pos[1])
    kernel_args_dynamic[7] = cp.float32(cam_pos[2])
    kernel_args_dynamic[17] = cp.float32(vx)
    kernel_args_dynamic[18] = cp.float32(vy)
    kernel_args_dynamic[19] = cp.float32(vz)
    kernel_args_dynamic[24] = cp.float32(focal_length)
    kernel_args_dynamic[-1] = cp.int32(frame_idx)  # frames
    cp.cuda.profiler.start()
    with render_stream:
        # 1. 核心光追
        trace_rays_kernel((grid_x, grid_y), (block_x, block_y), tuple(kernel_args_dynamic))
        # 2. 紧接着执行后处理 (GPU 会自动等光追画完再处理 Graph，绝不抢跑)
        postprocess_graph.launch(stream=render_stream)
        # debug((grid_x, grid_y), (block_x, block_y),(current_frame_float,w,h,frame_inter_tex.ptr))

    # ====== 【修复点 3】必须同步！让 CPU 停下来等 GPU 把所有活干完再截图 ======
    cp.cuda.profiler.stop()
    render_stream.synchronize()

    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
    # img_bgr_np[502:612,512:522,0]=255
    # img_bgr_np[602,512,1]=0
    # img_bgr_np[602,512,2]=255
    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(output_path, img_bgr_np)

    elapsed = time.time() - start_time
    t_val+=0.15902818

    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] {elapsed:.3f} s | "
          f"tau={tau:.1f} r={r:.4f} t={t_val:.2f} beta={np.sqrt(vx**2+vy**2+vz**2):.4f}")


    gc.collect()

print("\n渲染完毕！")
