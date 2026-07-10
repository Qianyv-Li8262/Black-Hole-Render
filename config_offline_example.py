"""
离线渲染配置文件(示例)
复制为 config_offline.py 并修改为你自己的参数,该文件已被 .gitignore 忽略。
"""

import numpy as np

# ========== 分辨率 ==========
w, h = 4096, 2160

# ========== CUDA 线程块 ==========
block_x, block_y = 32, 8

# ========== 传感器像素间距 (mm) ==========
# physwidth = w * PIXEL_PITCH, physheight = h * PIXEL_PITCH
PIXEL_PITCH = 0.002

# ========== 相机位置 ==========
cam_pos_init = np.array([21.81, -77.54, 14.45], dtype=np.float32)

# ========== 相机朝向 ==========
cam_yaw   = -11.49
cam_pitch = -0.38
cam_roll  = -0.2

# ========== 焦距 ==========
focal_length = 6

# ========== 光线步进 ==========
step    = 0.02   # 步长
maxstep = 4000   # 最大步数

# ========== 超采样 ==========
SSAA_COUNT = 256  # 每像素采样数

# ========== 模拟时间 ==========
start_t      = 20.5   # 起始坐标时
total_frames = 1      # 总帧数
d_tau        = 0.1    # 固有时间步长(解析下落模型)

# ========== 多普勒 / 相机速度 (beta) ==========
vx, vy, vz = 0.1574, 0.5873, 0

# ========== 文件路径(相对于项目根目录) ==========
skybox_path         = 'starmap_random_2020_16k.exr'
prebaked_disk_path  = 'prebaked_disk_noise_npgs.npy'
color_lut_path      = 'color_lut2.npy'
kernel_path         = 'blackholekernel3_prebaked copy.cu'
bloom_kernel_path   = 'postprocess_downup copy.cu'
output_dir          = 'output_frames'

# ========== 编译开关 ==========
# USE_RK4: 开启后渲染速度提升4-5倍,但吸积盘风格有差异,且存在采样伪影
USE_RK4           = False
NO_DEPTH_JITTER   = True
RAND_SAMP_DISK    = True
NO_BKGD_DOPPLER   = False