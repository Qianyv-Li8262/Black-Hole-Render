# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""
离线渲染配置文件(示例)
复制为 cfg/offline.py 并修改为你自己的参数,该文件已被 .gitignore 忽略。
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
d_tau        = 0.1    # render_single: 每帧相机固有时间间隔；坐标时由自由落体同步推进

# ========== 相机初速 (beta=v/c，依次沿相机前/右/上方向) ==========
# render_single 从 cam_pos_init 以此初速自由落体；全为0表示从静止释放。
vx, vy, vz = 0.1574, 0.5873, 0

# ========== 文件路径(相对于项目根目录) ==========
skybox_path         = 'assets/starmap_random_2020_16k.exr'
prebaked_disk_path  = 'cache/disk_native34_coefficients.npy'
# 用 python tools/prebake_disk_noise_npgs.py 生成旧厚云团后，可切换为
# 同时设置 DISK_COEFFICIENTS=False、OPACITY_CHANGE=True
# prebaked_disk_path = 'cache/prebaked_disk_clouds.npy'
color_lut_path      = 'cache/color_lut2.npy'
kernel_path         = 'krnls/disk.cu'
bloom_kernel_path   = 'krnls/bloom.cu'
output_dir          = 'output_frames'

# ========== 编译开关 ==========
# USE_RK4: 开启后渲染速度提升4-5倍,但吸积盘风格有差异,且存在采样伪影
USE_RK4           = False
NO_DEPTH_JITTER   = True
RAND_SAMP_DISK    = True
NO_BKGD_DOPPLER   = False
OPACITY_CHANGE    = False  # 旧材质选项；DISK_COEFFICIENTS=True 时由新缓存直接提供消光
USE_ACES          = True
NOT_USE_S_CURVE   = False

# 3+4+5消光 + main温度扰动；修改 cfg/disk_material.json 后重新烘焙
DISK_COEFFICIENTS = True
SKYBOX_SCALE = 330.0  # 原离线星空曝光33的10倍
