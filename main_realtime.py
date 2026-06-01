'''
This script is only for finding a good position to make pictures.
The accretion disk here is axissymmetric,which is a legacy edition that can give a good real-time performance.
I don't know the way to do temporal antialiasing(I'm a noob!)when the disk is moving so I give up real-time rendering and turned to offline rendering.
'''

import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
import glfw
from cupyx.scipy.ndimage import gaussian_filter
import os,sys
from zero_copy_window import ZeroCopyWindow
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
def create_texture_object(img_cp,num_of_channels):
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


base_path = os.path.dirname(os.path.abspath(__file__))
# img_file_path = os.path.join(base_path, 'eso0932a.tif')#改图片
# img_bgr = cv2.imread(img_file_path)   # 正常读取
img_file_path = os.path.join(base_path, 'starmap_random_2020_16k.exr') # 16k渲染视频
img_bgr = cv2.imread(img_file_path, cv2.IMREAD_UNCHANGED)   # exr读取




if img_bgr is None:
    print(f"错误：无法在路径 {img_file_path} 找到背景图片！")
    print("请检查图片文件名是否正确，或者图片是否在文件夹中。")
    exit() 

# img_bgr = cv2.imread(img_file_path)


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)*100
# img_float = img_rgb.astype(np.float32) / 255.0 # 正常读取
img_float = img_rgb.astype(np.float32) # exr读取
img_cp = cp.array(img_float)
# img_cp = gaussian_filter(img_cp, sigma=0.8, axes=(0, 1)) 
tex_handle, _internal_storage = create_texture_object(img_cp,3)

print('正在加载预烘焙吸积盘纹理...')
prebaked_data = np.load(os.path.join(base_path, 'prebaked_disk_noise.npy'))
ishalf=True
tex_prebaked = create_3d_texture_from_npy(cp.asarray(prebaked_data, dtype=cp.float16),ishalf)
print(f"  数据 shape: {prebaked_data.shape}  dtype: {'half' if ishalf else 'float32'}")
del prebaked_data
print('  吸积盘 3D 纹理就绪')

colorlut_file_path = os.path.join(base_path, 'color_lut2.npy')
lut_color= cp.load(colorlut_file_path).astype(cp.float32)

tex_handle_color,____=create_texture_object(lut_color,3)

kernel_path = os.path.join(base_path, "blackholekernel3_prebaked.cu") # 改为正常吸积盘渲染，这里改一下kernel
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

include_dir = os.path.dirname(os.path.abspath(kernel_path)) 
module = cp.RawModule(code=cuda_source, options=('-use_fast_math',"-I",include_dir))


trace_rays_kernel = module.get_function("blackholekernel")
# taa = module.get_function("taaColorClampingKernel")



kernel_path = os.path.join(base_path, "postprocess_gemini.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
bloom_module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
extract_bright_kernel = bloom_module.get_function("extract_bright_kernel")
blur_x_kernel = bloom_module.get_function("blur_x_kernel")
blur_y_fuse_kernel = bloom_module.get_function("blur_y_fuse_postprocess_kernel")

print('kernel complied')



# 超参数！


w,h=3200,2000

cam_pos = np.array([ -5.307298 , -10.113188  ,  1.5111736], dtype=np.float32)
cam_yaw = -4.82
cam_pitch = -0.25
cam_roll = 0.0 
focal_length = 1


move_speed = 0.05
turn_speed = 0.01
focus_speed=1.02
jitnum=1






window=ZeroCopyWindow(w,h,'try')
frame_intermediate_result=cp.empty((h * w * 4), dtype=cp.float32)
# accum1=cp.zeros((h * w * 4), dtype=cp.float32)
# accum2=cp.zeros((h * w * 4), dtype=cp.float32)
accum=cp.zeros((h * w * 4), dtype=cp.float32)
bright_buf = cp.empty((h * w * 4), dtype=cp.float32)
blur_x_tmp = cp.empty((h * w * 4), dtype=cp.float32)
bloom_threshold = np.float32(1.7)  # 超过多亮的区域产生光晕
bloom_radius = np.int32(20)        # 模糊采样半径 (越大光晕越宽)
bloom_sigma = np.float32(8.0)      # 高斯分布的平滑度
bloom_strength = np.float32(1.5)   # 光晕强度
block_x,block_y=32,8
grid_x=w//block_x+1 if w%block_x!=0 else w//block_x
grid_y=h//block_y+1 if h%block_y!=0 else h//block_y
print(grid_x)
tot_pixels=w*h
frames=1
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

a=0.04
t=15
while not window.should_close():
    current_frame_float = window.map_pbo()
    # th+=0.001
    # cam_pos = np.array([r*np.cos(th),r*np.sin(th), 0.0], dtype=np.float32)
    camera_moved = True
    
    if glfw.KEY_W in window.key_pressed:
        cam_pos += fwd * move_speed
        # focal_length=a*np.sqrt(cam_pos[0]**2-1)
        camera_moved = True
    if glfw.KEY_S in window.key_pressed:
        cam_pos -= fwd * move_speed
        # focal_length=a*np.sqrt(cam_pos[0]**2-1)
        camera_moved = True
    if glfw.KEY_D in window.key_pressed:
        cam_pos += right * move_speed
        camera_moved = True
    if glfw.KEY_A in window.key_pressed:
        cam_pos -= right * move_speed
        camera_moved = True
    if glfw.KEY_UP in window.key_pressed:
        cam_pos += up * move_speed 
        camera_moved = True
    if glfw.KEY_DOWN in window.key_pressed:
        cam_pos -= up * move_speed
        camera_moved = True
    if glfw.KEY_E in window.key_pressed:
        cam_yaw -= turn_speed
        camera_moved = True
    if glfw.KEY_Q in window.key_pressed:
        cam_yaw += turn_speed
        camera_moved = True
    if glfw.KEY_R in window.key_pressed: 
        cam_pitch += turn_speed
        camera_moved = True
    if glfw.KEY_F in window.key_pressed:  
        cam_pitch -= turn_speed
        camera_moved = True
    if glfw.KEY_Z in window.key_pressed:  
        cam_roll -= turn_speed
        camera_moved = True
    if glfw.KEY_C in window.key_pressed: 
        cam_roll += turn_speed
        camera_moved = True
    if glfw.KEY_G in window.key_pressed: 
        focal_length /= focus_speed
        camera_moved = True
    if glfw.KEY_T in window.key_pressed: 
        focal_length *= focus_speed
        camera_moved = True
    if camera_moved:
        accum.fill(0)
        frames = 1
        cam_pitch = np.clip(cam_pitch, -np.pi/2 + 0.001, np.pi/2 - 0.001)
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

    trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
        (frame_intermediate_result, cp.uint64(tex_handle.ptr),cp.uint64(tex_prebaked.ptr),cp.uint64(tex_handle_color.ptr),cp.float32(t),#改为正常吸积盘这里删去t
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),cp.float32(0), cp.float32(0), cp.float32(0),
         cp.int32(w), cp.int32(h),
         cp.float32(3.2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1), cp.int32(2000), cp.int32(jitnum),cp.int32(frames)))
    
    accum = accum + frame_intermediate_result
    # taa((grid_x, grid_y,), (block_x, block_y,),(frame_intermediate_result,accum1,accum2,cp.int32(w), cp.int32(h),cp.float32(0.1),cp.int32(frames)))
    # accum1,accum2 = accum2,accum1
    t+=0.5
    extract_bright_kernel((grid_x, grid_y), (block_x, block_y), 
                          (accum, bright_buf, np.int32(w), np.int32(h), 
                           np.float32(frames), bloom_threshold))
    blur_x_kernel((grid_x, grid_y), (block_x, block_y),
                  (bright_buf, blur_x_tmp, np.int32(w), np.int32(h), 
                   bloom_radius, bloom_sigma))
    blur_y_fuse_kernel((grid_x, grid_y), (block_x, block_y),
                       (accum, blur_x_tmp, current_frame_float, 
                        np.int32(w), np.int32(h), 
                        bloom_radius, bloom_sigma, 
                        np.float32(frames), bloom_strength))
    
    frames += 1
    window.unmap_and_draw()
    # print(cam_pos)
    # print(cam_pitch)
    # print(cam_yaw)
    # print(focal_length)
    # if frames == 2:
    #     sys.exit()


window.destroy()
print('Done.')