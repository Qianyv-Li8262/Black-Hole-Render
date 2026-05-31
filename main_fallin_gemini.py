import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
import os, sys

def create_texture_object(img_cp, num_of_channels):
    h, w, c = img_cp.shape
    bytes_per_pixel = 16 
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :num_of_channels] = img_cp
    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D, 
        arr=rgba,                  
        chDesc=ch_fmt,  
        width=w,
        height=h,
        pitchInBytes=pitch_bytes
    )
    tex_ptr = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj, rgba

def create_texture_object_nopadding(img_cp_padded, num_of_channels,h_real,w_real,pitch_bytes,is_half = False):
    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat) if not is_half else texture.ChannelFormatDescriptor(16,16,16,16, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D, 
        arr=img_cp_padded,                  
        chDesc=ch_fmt,  
        width=w_real,
        height=h_real,
        pitchInBytes=pitch_bytes
    )
    tex_ptr = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj

# ================= 广义相对论测地线解析物理求解器（摆线反解法） =================
def update_camera_physics_analytical(tau, r_start, dir_unit, fwd, right, up, d_tau=1.0):
    """
    计算从无穷远处静止释放的相机，在下落到本征时间 tau 时的严格解析物理状态。
    相机在 tau = 0 时处于指定的有限半径 r_start 处（带有初速度）。
    """
    M = 1.0
    # 1. 计算初始位置 r_start 对应的史瓦西半径 R_start
    R_start = r_start + 1.0 + 0.25 / r_start
    
    # 2. 计算落入奇点 (R = 0) 的最大本征时间限制 tau_max
    # tau_max = 2/3 * R_start^1.5 / sqrt(2) = sqrt(2)/3 * R_start^1.5
    tau_max = (np.sqrt(2.0) / 3.0) * (R_start**1.5)
    
    if tau >= tau_max:
        # 如果已经落入奇点，让相机停留在事件视界外侧 r = 0.51 处
        r_next = 0.51
        beta_mag = 0.999
    else:
        # 3. 严格解析闭合解：求当前本征时 tau 对应的史瓦西半径 R
        # 物理上完全不需要任何数值迭代，一行代码搞定！
        R = (R_start**1.5 - (1.5 * np.sqrt(2.0)) * tau) ** (2.0 / 3.0)
        
        # 规避数值溢出，限制其不低于视界边缘 2.0001
        R = max(2.0001, R)
        
        # 4. 将标准史瓦西 R 转换回各向同性坐标 r_next
        r_next = 0.5 * ((R - 1.0) + np.sqrt((R - 1.0)**2 - 1.0))
        
        # 5. 严格计算从无穷远落下的局域物理速度 beta
        u_next = 1.0 / (2.0 * r_next)
        beta_mag = np.sqrt(2.0 / r_next) / (1.0 + u_next)
        beta_mag = min(0.999, beta_mag)
        
    # 6. 全局速度向量 (径向自由下落，方向指向黑洞)
    beta_global = -beta_mag * dir_unit
    
    # 7. 将全局速度投影到相机的局部坐标系中
    vx_next = np.dot(beta_global, fwd)
    vy_next = np.dot(beta_global, right)
    vz_next = np.dot(beta_global, up)
    
    # 8. 计算这一帧由时间膨胀折算出的坐标时步长 dt = factor * gamma * d_tau
    u_next = 1.0 / (2.0 * r_next)
    factor = (1.0 + u_next) / (1.0 - u_next)
    gamma = 1.0 / np.sqrt(1.0 - beta_mag**2)
    dt = factor * gamma * d_tau
    
    return r_next, vx_next, vy_next, vz_next, dt

base_path = os.path.dirname(os.path.abspath(__file__))
img_file_path = os.path.join(base_path, 'starmap_random_2020_16k.exr')
print('Start texture creating.')
img_bgr = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)

if img_bgr is None:
    print(f"错误：无法在路径 {img_file_path} 找到背景图片！喵")
    exit() 

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)*300
del img_bgr
img_float = img_rgb.astype(np.float16)
# img_float = img_rgb.astype(np.float16) / 255.0 # 正常读取
del img_rgb
hh,ww,cc = img_float.shape
bytes_per_pixel = 8 
alignment = 256
pitch_bytes = ((ww * bytes_per_pixel + alignment - 1) // alignment) * alignment
padded_w = pitch_bytes // bytes_per_pixel
rgba = np.zeros((hh, padded_w, 4), dtype=np.float16)
rgba[:, :ww, :3] = img_float
img_cp = cp.array(rgba)
del img_float
del rgba
tex_handle = create_texture_object_nopadding(img_cp,3,hh,ww,pitch_bytes,True)
print('Texture created successfully,starting kernel compilation and LUT initializing.')

physlut_file_path = os.path.join(base_path, 'disk_lut_for_mov_disk.npy')
lut_phys = cp.load(physlut_file_path).astype(cp.float32)
tex_handle_lut, ____ = create_texture_object(lut_phys, 4)

colorlut_file_path = os.path.join(base_path, 'color_lut2.npy')
lut_color = cp.load(colorlut_file_path).astype(cp.float32)
tex_handle_color, ____ = create_texture_object(lut_color, 3)

kernel_path = os.path.join(base_path, "blackholekernel3_moving disk speed.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

include_dir = os.path.dirname(os.path.abspath(kernel_path)) 
module = cp.RawModule(code=cuda_source, options=('-use_fast_math', "-I", include_dir))
trace_rays_kernel = module.get_function("blackholekernel")

kernel_path = os.path.join(base_path, "postprocess_gemini.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
bloom_module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
extract_bright_kernel = bloom_module.get_function("extract_bright_kernel")
blur_x_kernel = bloom_module.get_function("blur_x_kernel")
blur_y_fuse_kernel = bloom_module.get_function("blur_y_fuse_postprocess_kernel")

print('Kernels compiled.')

# ================= 超参数配置区域  =================
w, h = 3200,2000
# 离线视频输出设置 
total_frames = 100            # 需要导出的总帧数
start_t = 75                  # 起始坐标时间 t

# 64x SSAA 离线画质
SSAA_COUNT =    16               

# 专用输出文件夹
output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)

# ================= 物理参数初始化 =================
cam_pos_init = np.array([20.0, 0.0, 3], dtype=np.float32)
r0 = np.linalg.norm(cam_pos_init) # 初始释放位置
dir_unit = cam_pos_init / r0      # 下落的径向方向单位向量

# 物理状态量初始化
r = r0
t = start_t
tau = 0.0                         # 本征时计数器初始化
vx, vy, vz = 0.1, -0.3, 0.0        # 初始速度为 0
d_tau = 1.0                       # 设定每两帧之间的本征时步长为 1.0

cam_yaw = -3.14
cam_pitch = -0.1488899
cam_roll = 0.333 
focal_length = 1

# 缓存分配
frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
blur_x_tmp = cp.empty((h * w * 4), dtype=cp.float32)
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8)

bloom_threshold = np.float32(1.7)  
bloom_radius = np.int32(20)        
bloom_sigma = np.float32(8.0)      
bloom_strength = np.float32(1.5)   

block_x, block_y = 32, 8
grid_x = w // block_x + 1 if w % block_x != 0 else w // block_x
grid_y = h // block_y + 1 if h % block_y != 0 else h // block_y

world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
fwd_z = np.sin(cam_pitch)
fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
fwd /= np.linalg.norm(fwd)
right0 = np.cross(fwd, world_up)
right_norm = np.linalg.norm(right0)
if right_norm > 1e-6:
    right0 /= right_norm
else:
    right0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
up0 = np.cross(right0, fwd)
up0 /= np.linalg.norm(up0)
right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)

# ================= 离线渲染循环 =================
print(f"开始离线渲染,总计 {total_frames} 帧，输出目录: {output_dir}")

cam_pos = r * dir_unit
start_time = time.time()
    
# cp.cuda.profiler.start()
# 2. 光线追踪核心 (传入动态计算的 t, cam_pos 和投影后的速度 vx, vy, vz)
trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
        (frame_intermediate_result, cp.uint64(tex_handle.ptr), cp.uint64(tex_handle_lut.ptr), cp.uint64(tex_handle_color.ptr), cp.float32(t),
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
         cp.float32(vx), cp.float32(vy), cp.float32(vz), 
         cp.int32(w), cp.int32(h),
         cp.float32(3.2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1), cp.int32(2000), cp.int32(16), cp.int32(1)))
# cp.cuda.profiler.stop()

extract_bright_kernel((grid_x, grid_y), (block_x, block_y), 
                          (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h), 
                           np.float32(1), bloom_threshold))
    
blur_x_kernel((grid_x, grid_y), (block_x, block_y),
                  (bright_buf, blur_x_tmp, np.int32(w), np.int32(h), 
                   bloom_radius, bloom_sigma))
    
blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
                       (frame_intermediate_result, blur_x_tmp, current_frame_float, 
                        np.int32(w), np.int32(h), 
                        bloom_radius, bloom_sigma, 
                        np.float32(1), bloom_strength))
    
    # 等待 GPU 同步
cp.cuda.Device().synchronize()
    
    # 3. 数据转换
img_rgba_cp = current_frame_float.reshape((h, w, 4))
img_rgba_np = cp.asnumpy(img_rgba_cp)
img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
    

end_time = time.time()
elapsed = end_time - start_time
print(f'预渲染完成，预计每帧用时{elapsed/16*SSAA_COUNT:.3f} s，预计总用时：{(xx:=elapsed/16*SSAA_COUNT*total_frames):.3f} s, \n共计{xx//3600} hr {((xx%3600)//60)} min {xx%60:.3f} secs')
for frame_idx in range(1, total_frames + 1):
    # 1. 基于当前物理状态生成相机位置
    cam_pos = r * dir_unit
    start_time = time.time()
    
    cp.cuda.profiler.start()
    # 2. 光线追踪核心 (传入动态计算的 t, cam_pos 和投影后的速度 vx, vy, vz)
    trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
        (frame_intermediate_result, cp.uint64(tex_handle.ptr), cp.uint64(tex_handle_lut.ptr), cp.uint64(tex_handle_color.ptr), cp.float32(t),
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
         cp.float32(vx), cp.float32(vy), cp.float32(vz), 
         cp.int32(w), cp.int32(h),
         cp.float32(3.2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1), cp.int32(2000), cp.int32(SSAA_COUNT), cp.int32(frame_idx)))
    cp.cuda.profiler.stop()

    extract_bright_kernel((grid_x, grid_y), (block_x, block_y), 
                          (frame_intermediate_result, bright_buf, np.int32(w), np.int32(h), 
                           np.float32(1), bloom_threshold))
    
    blur_x_kernel((grid_x, grid_y), (block_x, block_y),
                  (bright_buf, blur_x_tmp, np.int32(w), np.int32(h), 
                   bloom_radius, bloom_sigma))
    
    blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
                       (frame_intermediate_result, blur_x_tmp, current_frame_float, 
                        np.int32(w), np.int32(h), 
                        bloom_radius, bloom_sigma, 
                        np.float32(1), bloom_strength))
    
    # 等待 GPU 同步
    cp.cuda.Device().synchronize()
    
    # 3. 数据转换
    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
    
    file_name = f"frame_{frame_idx:05d}.png"
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, img_bgr_np)

    end_time = time.time()
    elapsed = end_time - start_time
    
    # 4. 【解析测地线更新】计算并更新下一帧所需的物理量
    if frame_idx < total_frames:
        tau += d_tau # 均匀累加本征时间步长
        r, vx, vy, vz, dt = update_camera_physics_analytical(tau, r0, dir_unit, fwd, right, up, d_tau)
        t += dt     # 累加由引力和运动时间膨胀换算出的当前坐标时步长 dt

    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] 渲染完成 | 耗时: {elapsed:.3f} 秒 | 本征时 tau = {tau:.1f} | 坐标位置 r = {r:.4f} | 坐标时 t = {t:.2f} | 局部物理速度 beta = {np.sqrt(vx**2+vy**2+vz**2):.4f}")

print("\n所有帧渲染完毕喵！")