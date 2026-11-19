# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Copy to cfg/offline.py in a clean checkout to run the 1080p example."""
from math import atan2, hypot

from cfg.offline_example import *


DISK_LOOP_PERIOD = 0.0  # Set to 40.0 for the three-frame closure check.

w, h = 1920, 1080
PIXEL_PITCH = 0.004
SSAA_COUNT = 4
step = 0.04
maxstep = 3000
focal_length = 5.4

cam_pos_init = np.array([28.0, -44.0, 9.0], dtype=np.float64)
_x, _y, _z = cam_pos_init
cam_yaw = atan2(-_y, -_x)
cam_pitch = atan2(-_z, hypot(_x, _y))
cam_roll = 0.10
CAMERA_MOTION = 'static'
vx = vy = vz = 0.0
start_t = 7.0

skybox_path = 'assets/starmap_random_2020_4k.exr'
SKYBOX_SCALE = 20.0
prebaked_disk_path = 'output_frames/repro_1080p/cache/disk_a.npy'
DISK_LOOP_TEXTURE_B = 'output_frames/repro_1080p/cache/disk_b.npy'

if DISK_LOOP_PERIOD > 0:
    # Sample t, t+T/2 and t+T. Convert coordinate-time spacing to proper time.
    _radius = hypot(*cam_pos_init)
    d_tau = (DISK_LOOP_PERIOD / 2) * (_radius - 0.5) / (_radius + 0.5)
    total_frames = 3
    output_dir = 'output_frames/repro_1080p/loop'
else:
    total_frames = 1
    output_dir = 'output_frames/repro_1080p/still'
