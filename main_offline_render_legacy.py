import numpy as np
import cupy as cp
import cv2
import time
from cuda_tex import create_texture_array_2d
from cupyx.scipy.ndimage import gaussian_filter
import os, sys
base_path = os.path.dirname(os.path.abspath(__file__))
# img_file_path = os.path.join(base_path, 'eso0932a.tif')
img_file_path = os.path.join(base_path, 'starmap_random_2020_16k.exr')
print('Start texture creating.')
img_bgr = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)   # exr读取
# img_bgr = cv2.imread(img_file_path)

if img_bgr is None:
    print(f"错误:无法在路径 {img_file_path} 找到背景图片!喵")
    exit() 

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)*30
del img_bgr
img_float = cp.asarray(img_rgb.astype(np.float16))
del img_rgb
skybox_rgba = cp.zeros((*img_float.shape[:2], 4), dtype=cp.float16)
skybox_rgba[:, :, :3] = img_float
del img_float
tex_handle = create_texture_array_2d(skybox_rgba, 4, (1, 1, 1, 1), is_half=True)
del skybox_rgba
print('Texture created successfully,starting kernel compilation and LUT initializing.')
# tex_handle, _internal_storage = create_texture_object(img_cp, 3)

physlut_file_path = os.path.join(base_path, 'disk_lut_for_mov_disk.npy')
lut_phys = cp.load(physlut_file_path).astype(cp.float32)
tex_handle_lut = create_texture_array_2d(lut_phys, 4, (1, 1, 1, 1))

colorlut_file_path = os.path.join(base_path, 'color_lut2.npy')
lut_color = cp.load(colorlut_file_path).astype(cp.float32)
lut_rgba = cp.zeros((*lut_color.shape[:2], 4), dtype=cp.float32)
lut_rgba[:, :, :3] = lut_color
tex_handle_color = create_texture_array_2d(lut_rgba, 4, (1, 1, 1, 1))

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
vx,vy,vz=0,0,0
# 离线视频输出设置 
total_frames = 200             # 需要导出的总帧数
delta_t = 0.25                   # 物理时间 t 步长
start_t = 75                   # 起始物理时间 

# 64x SSAA 离线画质拉满 喵,可以再增大 喵
SSAA_COUNT = 16                  

# 专用输出文件夹
output_dir = os.path.join(base_path, 'output_frames')
os.makedirs(output_dir, exist_ok=True)

cam_pos = np.array([ 20  , 0  ,   3], dtype=np.float32)
cam_yaw = -3.14
cam_pitch = -0.25
cam_roll = 0.0 
focal_length = 1

# 缓存分配(不再使用 PBO,直接在显存分配数组)
frame_intermediate_result = cp.empty((h * w * 4), dtype=cp.float32)
bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
blur_x_tmp = cp.empty((h * w * 4), dtype=cp.float32)
current_frame_float = cp.empty((h * w * 4), dtype=cp.uint8) # 取代 window.map_pbo()

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
print(f"开始离线渲染,总计 {total_frames} 帧,输出目录: {output_dir}")

for frame_idx in range(1, total_frames + 1):
    t = start_t + (frame_idx - 1) * delta_t
    start_time = time.time()
    cp.cuda.profiler.start()
    # 1. 光线追踪核心(开启 64x SSAA 喵)
    trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
        (frame_intermediate_result, cp.uint64(tex_handle.ptr), cp.uint64(tex_handle_lut.ptr), cp.uint64(tex_handle_color.ptr), cp.float32(t),
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),cp.float32(vx),cp.float32(vy),cp.float32(vz),
         cp.int32(w), cp.int32(h),
         cp.float32(3.2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1), cp.int32(2000), cp.int32(SSAA_COUNT), cp.int32(frame_idx)))
    cp.cuda.profiler.stop()
    # frame_intermediate_result = frame_intermediate_result * frame_idx
    # 2. Bloom 泛光后处理
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
    
    # 等待 GPU 同步计算,确保数据落盘前计算已完成
    cp.cuda.Device().synchronize()
    
    # 3. 数据转换:RGBA Float32 转换为 RGBA 8-bit
    img_rgba_cp = current_frame_float.reshape((h, w, 4))
    # img_rgba_cp = cp.clip(img_rgba_cp * 255.0, 0, 255).astype(cp.uint8)
    
    # 4. 拷贝到 CPU 并使用 OpenCV 保存为 PNG
    img_rgba_np = cp.asnumpy(img_rgba_cp)
    img_bgr_np = cv2.cvtColor(img_rgba_np, cv2.COLOR_RGBA2BGR)
    
    file_name = f"frame_{frame_idx:05d}.png"
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, img_bgr_np)
    
    end_time = time.time()
    elapsed = end_time - start_time
    vx+=0.02
    # cam_pos += fwd * 0.075
    print(f"[Frame {frame_idx:05d}/{total_frames:05d}] 渲染完成 | 耗时: {elapsed:.3f} 秒 | 物理时间 t = {t:.2f}")
print("\n所有帧渲染完毕喵!")
print("你可以打开终端,进入输出目录运行以下 ffmpeg 命令拼成精美的 MP4 视频:")
print(f"cd /D {output_dir}")
print("ffmpeg -r 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")