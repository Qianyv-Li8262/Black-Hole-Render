import numpy as np
import cupy as cp
import cv2
import time
from cuda_tex import create_texture_array_2d, create_texture_array_3d
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
img_bgr = cv2.imread(os.path.join(base_path, 'starmap_random_2020_16k.exr'),
                     cv2.IMREAD_UNCHANGED)
# img_bgr = cv2.imread(os.path.join(base_path, 'black.bmp'),
                    #  cv2.IMREAD_UNCHANGED)
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
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise.npy'))
ishalf=True
tex_prebaked = create_texture_array_3d(cp.asarray(prebaked_data, dtype=cp.float16),4,(0,0,0,1,1),ishalf)
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
kernel_path = os.path.join(base_path, "blackholekernel3_prebaked.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
module = cp.RawModule(code=cuda_source, options=('-use_fast_math','-lineinfo'))
trace_rays_kernel = module.get_function("blackholekernel")


bloom_path = os.path.join(base_path, "postprocess_downup.cu")
with open(bloom_path, "r", encoding="utf-8") as f:
    bloom_source = f.read()
bloom_module = cp.RawModule(code=bloom_source, options=('-use_fast_math',))
gaussH = bloom_module.get_function("gaussianBlurH")
gaussW = bloom_module.get_function("gaussianBlurW")
bloom = bloom_module.get_function("compositeBloom")
bright = bloom_module.get_function("extractBright")
downsample2x = bloom_module.get_function("downsample2x")
print('  Kernel 编译完成')


w, h = 4096,2160
total_frames = 10
start_t = 50
SSAA_COUNT = 1024

output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)

cam_pos_init = np.array([  2.913251 ,  -11.766628  ,   1.05188831], dtype=np.float32)
r0 = np.linalg.norm(cam_pos_init)
dir_unit = cam_pos_init / r0

r = r0
t_val = start_t
tau = 0.0
d_tau = 0.1

cam_yaw, cam_pitch, cam_roll = -4.76, -0.05, -0
focal_length = 1.9

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
w2,h2=w//2,h//2
w4,h4=w//4,h//4
w8,h8=w//8,h//8
w16,h16=w//16,h//16
w32,h32=w//32,h//32
w64,h64=w//64,h//64
w128,h128=w//128,h//128
w256,h256=w//256,h//256

# _, vx, vy, vz, _ = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)


frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
down1=cp.zeros((h2*w2*4),dtype=cp.float32)
down2=cp.zeros((h4*w4*4),dtype=cp.float32)
down3=cp.zeros((h8*w8*4),dtype=cp.float32)
down4=cp.zeros((h16*w16*4),dtype=cp.float32)
down5=cp.zeros((h32*w32*4),dtype=cp.float32)
down6=cp.zeros((h64*w64*4),dtype=cp.float32)
down7=cp.zeros((h128*w128*4),dtype=cp.float32)
down8=cp.zeros((h256*w256*4),dtype=cp.float32)
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

bloom_threshold = np.float32(0.5)
bloom_radius = np.int32(20)
bloom_sigma = np.float32(8.0)
bloom_strength = np.float32(1.5)

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
    cp.float32(8.192), cp.float32(4.320),        # physwidth, physheight
    cp.float32(focal_length),
    cp.float32(0.1),                        # step
    cp.int32(2000),                         # maxstep
    cp.int32(SSAA_COUNT),                   # jitternum
    cp.int32(1),                            # frames
)

bright_buf = cp.zeros_like(frame_intermediate_result)
tmp_blur_buf = cp.empty((h * w * 4), dtype=cp.float32)
print(f"\n开始离线渲染, 总计 {total_frames} 帧, 输出目录: {output_dir}")
def make_downsample(input_tex_ptr, out_buf, out_w, out_h, tex_list):
    grid_out_x = (out_w + block_x - 1) // block_x
    grid_out_y = (out_h + block_y - 1) // block_y
    downsample2x((grid_out_x, grid_out_y), (block_x, block_y), 
                 (cp.uint64(input_tex_ptr), out_buf, cp.int32(out_w), cp.int32(out_h)))
    tex_down = create_texture_array_2d(out_buf.reshape(out_h, out_w, 4), 4, (1,1,1,1))
    tex_list.append(tex_down)
    tmp_slice = tmp_blur_buf[:out_w * out_h * 4]
    gaussH((grid_out_x, grid_out_y), (block_x, block_y), 
           (tmp_slice, cp.int32(out_w), cp.int32(out_h), cp.uint64(tex_down.ptr)))
    tex_tmp = create_texture_array_2d(tmp_slice.reshape(out_h, out_w, 4), 4, (1,1,1,1))
    tex_list.append(tex_tmp)
    gaussW((grid_out_x, grid_out_y), (block_x, block_y), 
           (out_buf, cp.int32(out_w), cp.int32(out_h), cp.uint64(tex_tmp.ptr)))
    ret_tex = create_texture_array_2d(out_buf.reshape(out_h, out_w, 4), 4, (1,1,1,1))
    tex_list.append(ret_tex)
    return ret_tex
for frame_idx in range(1, total_frames + 1):
    cam_pos = r * dir_unit
    start_time = time.time()
    temp_textures=[]

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
    bright((grid_x, grid_y), (block_x, block_y),(frame_intermediate_result,bright_buf,cp.int32(w),cp.int32(h),cp.float32(1.0),cp.float32(bloom_threshold)))
    tex_bright= create_texture_array_2d(bright_buf.reshape(h,w,4),4,(1,1,1,1))
    temp_textures.append(tex_bright)
    tex_down1 = make_downsample(tex_bright.ptr, down1, w2, h2, temp_textures)
    tex_down2 = make_downsample(tex_down1.ptr,  down2, w4, h4, temp_textures)   
    tex_down3 = make_downsample(tex_down2.ptr,  down3, w8, h8, temp_textures)
    tex_down4 = make_downsample(tex_down3.ptr,  down4, w16, h16, temp_textures)
    tex_down5 = make_downsample(tex_down4.ptr,  down5, w32, h32, temp_textures)
    tex_down6 = make_downsample(tex_down5.ptr,  down6, w64, h64, temp_textures)
    tex_down7 = make_downsample(tex_down6.ptr,  down7, w128, h128, temp_textures)
    tex_down8 = make_downsample(tex_down7.ptr,  down8, w256, h256, temp_textures)
    cp.cuda.Device().synchronize()
    tex_group=cp.array([tex_down1.ptr,tex_down2.ptr,tex_down3.ptr,tex_down4.ptr,tex_down5.ptr,tex_down6.ptr,tex_down7.ptr,tex_down8.ptr],dtype=cp.uint64)
    tex_original = create_texture_array_2d(frame_intermediate_result.reshape(h,w,4), 4, (1,1,1,1))
    temp_textures.append(tex_original)
    bloom((grid_x, grid_y), (block_x, block_y),(current_frame_float,cp.int32(w),cp.int32(h),cp.uint64(tex_original.ptr),tex_group,cp.int32(w),cp.int32(h)))


    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)

    output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(output_path, img_bgr_np)

    elapsed = time.time() - start_time
    t_val+=0.1


    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] {elapsed:.3f} s | "
          f"tau={tau:.1f} r={r:.4f} t={t_val:.2f} beta={np.sqrt(vx**2+vy**2+vz**2):.4f}")
    del tex_bright
    del tex_down1, tex_down2, tex_down3, tex_down4, tex_down5, tex_down6, tex_down7, tex_down8
    del tex_original
    del tex_group
    
    temp_textures.clear()

    gc.collect()

print("\n渲染完毕！")
