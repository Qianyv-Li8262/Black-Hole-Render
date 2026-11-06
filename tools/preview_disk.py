# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Render one disk preview (2K by default) without editing the offline config."""

from pathlib import Path
import argparse
import math
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=float, default=0.01)
    parser.add_argument('--jitter', type=int, choices=range(1, 65), default=8)
    parser.add_argument('--skybox', help='Skybox path relative to repository root')
    parser.add_argument('--disk', help='Prebaked disk path relative to repository root')
    parser.add_argument('--camera', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--yaw', type=float)
    parser.add_argument('--pitch', type=float)
    parser.add_argument('--roll', type=float)
    parser.add_argument('--velocity', type=float, nargs=3, metavar=('VX', 'VY', 'VZ'))
    parser.add_argument('--time', type=float, help='Override the coordinate time for material comparisons')
    parser.add_argument('--exposure-scale', type=float, default=1.0,
                        help='Scale linear HDR before the existing bloom/tone mapping; raw HDR saves stay unscaled')
    parser.add_argument('--width', type=int, default=2048, help='Preview width, preserving sensor size and aspect ratio')
    parser.add_argument('--output', default='output_frames/disk_soft_clouds_2k')
    parser.add_argument('--save-linear', action='store_true', help='Save linear HDR pixels for convergence checks')
    args = parser.parse_args()
    if not math.isfinite(args.step) or args.step <= 0:
        parser.error('--step must be positive and finite')
    if args.width < 32:
        parser.error('--width must be at least 32')
    if args.time is not None and not math.isfinite(args.time):
        parser.error('--time must be finite')
    if not math.isfinite(args.exposure_scale) or args.exposure_scale <= 0:
        parser.error('--exposure-scale must be positive and finite')
    if args.velocity is not None and (not all(math.isfinite(v) for v in args.velocity)
                                     or sum(v*v for v in args.velocity) >= 1):
        parser.error('--velocity must be finite with speed below 1')

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    # Must precede importing OpenCV in the renderer.
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    from cfg import offline as cfg

    # Preserve the configured sensor size / field of view at any preview width.
    sensor_width = cfg.w * cfg.PIXEL_PITCH
    aspect = cfg.h / cfg.w
    cfg.w, cfg.h = args.width, round(args.width * aspect)
    cfg.PIXEL_PITCH = sensor_width / cfg.w
    cfg.SSAA_COUNT = args.jitter
    cfg.step = args.step
    cfg.maxstep = max(cfg.maxstep, math.ceil(80.0 / args.step))
    cfg.total_frames = 1
    cfg.output_dir = args.output
    if args.time is not None:
        cfg.start_t = args.time
    if args.skybox:
        cfg.skybox_path = args.skybox
    if args.disk:
        cfg.prebaked_disk_path = args.disk
    if args.camera:
        import numpy as np

        cfg.cam_pos_init = np.array(args.camera, dtype=np.float32)
    if args.velocity is not None:
        cfg.vx, cfg.vy, cfg.vz = args.velocity
    for name in ('yaw', 'pitch', 'roll'):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, 'cam_' + name, value)
    rendered = runpy.run_path(str(root / 'render_single.py'), run_name='__main__',
                             init_globals={'_render_argv': []})
    if args.save_linear or args.exposure_scale != 1:
        import numpy as np

        pixels = np.empty((cfg.h, cfg.w, 4), dtype=np.float32)
        rendered['frame_inter_tex'].ResDesc.cuArr.copy_to(pixels.reshape(cfg.h, cfg.w * 4))
        if not np.isfinite(pixels).all():
            raise RuntimeError('Non-finite pixels in disk preview')
        if args.save_linear:
            np.save(root / cfg.output_dir / 'linear_rgba.npy', pixels)
        if args.exposure_scale != 1:
            import cupy as cp
            import cv2
            from helpers.render_manifest import finish_manifest

            scaled = cp.asarray(pixels)
            scaled[..., :3] *= args.exposure_scale
            rendered['frame_inter_tex'].ResDesc.cuArr.copy_from(scaled.reshape(cfg.h, cfg.w*4))
            cp.cuda.get_current_stream().synchronize()
            rendered['postprocess_graph'].launch(stream=rendered['render_stream'])
            rendered['render_stream'].synchronize()
            rgba = cp.asnumpy(rendered['current_frame_float'].reshape(cfg.h, cfg.w, 4))
            target = root / cfg.output_dir / 'frame_00001.png'
            if not cv2.imwrite(str(target), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)):
                raise OSError(f'Could not save preview: {target}')
            finish_manifest(rendered['manifest_path'], 'complete',
                            {'preview_exposure_scale': args.exposure_scale})


if __name__ == '__main__':
    main()
