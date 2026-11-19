# SPDX-License-Identifier: GPL-3.0-or-later
"""Check A/B rendering through the frame-parallel entry point."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import render_hybrid as hybrid

ROOT = Path(__file__).resolve().parents[1]
try:
    GPU_AVAILABLE = hybrid.cp is not None and hybrid.cp.cuda.runtime.getDeviceCount() > 0
except RuntimeError:
    GPU_AVAILABLE = False


class HybridLoopTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'temp').mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'temp')
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def test_two_workers_assign_frames_once_and_resume_with_loop_metadata(self):
        claims = []
        barrier = threading.Barrier(2)
        def worker(gpu_id, scheduler, time_step, output_dir):
            first = scheduler.claim()
            if first is None:
                return
            barrier.wait(timeout=10)
            frame = first
            while frame is not None:
                claims.append((gpu_id, frame, hybrid.cfg.start_t+(frame-1)*time_step))
                (output_dir / f'frame_{frame:05d}.png').write_bytes(b'rendered')
                scheduler.finish(frame)
                frame = scheduler.claim()
        cuda = MagicMock()
        cuda.cuda.runtime.getDeviceCount.return_value = 2
        with patch.multiple(hybrid.cfg, DISK_LOOP_PERIOD=40., DISK_COEFFICIENTS=True,
                            DISK_LOOP_TEXTURE_B='cache/disk_native34_b.npy',
                            kernel_path='krnls/disk.cu', output_dir=str(self.output),
                            total_frames=6, start_t=7.), \
                patch.object(hybrid, 'cp', cuda), patch.object(hybrid, 'worker', worker), \
                patch.object(sys, 'argv', ['render_hybrid.py', '--gpus', 'all', '--time-step', '.5']):
            self.assertEqual(hybrid.main(), 0)
            self.assertEqual(sorted(frame for _, frame, _ in claims), list(range(1, 7)))
            self.assertEqual({gpu for gpu, _, _ in claims}, {0, 1})
            for _, frame, time in claims:
                self.assertEqual(time, 7+(frame-1)*.5)
            manifest = json.loads(next(self.output.glob('run_*.json')).read_text())
            self.assertTrue(manifest['disk_loop']['enabled'])
            self.assertEqual(manifest['summary']['done'], 6)
            self.assertEqual(hybrid.main(), 0)
            self.assertEqual(len(claims), 6)

    def test_cpu_worker_is_rejected_before_initialization(self):
        with patch.multiple(hybrid.cfg, DISK_LOOP_PERIOD=40., DISK_COEFFICIENTS=True,
                            kernel_path='krnls/disk.cu'), \
                patch.object(sys, 'argv', ['render_hybrid.py', '--use-cpu', '--cpu-cores', '1']):
            with self.assertRaises(SystemExit) as error:
                hybrid.main()
            self.assertEqual(error.exception.code, 2)

    def test_skybox_rejects_png(self):
        sky = self.output / 'sky.png'
        self.assertTrue(cv2.imwrite(str(sky), np.zeros((2, 4, 3), np.uint8)))
        with self.assertRaisesRegex(ValueError, 'OpenEXR'):
            hybrid.read_skybox_bgr(sky)

    @unittest.skipUnless(GPU_AVAILABLE, 'CuPy and CUDA required')
    def test_actual_gpu_renderer_closes_out_of_order_loop_frames_and_falls_back(self):
        import OpenEXR

        sky = self.output / 'sky.exr'
        black = np.zeros((2, 4), np.float32)
        with OpenEXR.File({}, {channel: black for channel in 'RGB'}) as exr:
            exr.write(str(sky))
        lut = self.output / 'color.npy'
        np.save(lut, np.ones((2, 16, 3), np.float32))
        paths = []
        for seed, (name, emission) in enumerate([('a', .04), ('b', .14)]):
            path = self.output / (name+'.npy')
            texture = np.empty((8, 16, 32, 4), np.float32)
            texture[...] = (.02, 4500., emission, 0.)
            np.save(path, texture)
            path.with_suffix('.json').write_text(json.dumps(
                dict(format='native34-coefficients-v1', vertical_half_extent=3.5,
                     settings={'resolution': [8, 16, 32]}, noise_seed=seed,
                     physics_lut_sha256='test-physics', core_profile_sha256='test-core')))
            paths.append(str(path))
        settings = dict(w=16, h=16, block_x=8, block_y=8, skybox_path=str(sky),
                        SKYBOX_SCALE=1., prebaked_disk_path=paths[0], DISK_LOOP_TEXTURE_B=paths[1],
                        color_lut_path=str(lut), DISK_COEFFICIENTS=True, DISK_LOOP_PERIOD=40.,
                        DISK_LOOP_REFERENCE_TIME=20., DISK_LOOP_EPOCH=0., start_t=7., total_frames=3,
                        cam_pos_init=np.array([40., 0., 0.]), cam_yaw=np.pi, cam_pitch=0., cam_roll=0.,
                        vx=0., vy=0., vz=0., PIXEL_PITCH=.0001, focal_length=4., step=.01,
                        maxstep=10000, SSAA_COUNT=2, USE_RK4=False, NO_DEPTH_JITTER=False,
                        RAND_SAMP_DISK=True, NO_BKGD_DOPPLER=True,
                        kernel_path='krnls/disk.cu', bloom_kernel_path='krnls/bloom.cu')
        cp = hybrid.cp
        with patch.multiple(hybrid.cfg, **settings), cp.cuda.Device(0):
            renderer = hybrid.GpuRenderer(0, 20., self.output)
            def render(frame):
                renderer.render_frame(frame)
                hdr = np.empty((16, 16, 4), np.float32)
                renderer.frame_inter_tex.ResDesc.cuArr.copy_to(hdr.reshape(16, 64))
                self.assertTrue(np.isfinite(hdr).all())
                return hdr
            # Simulate arbitrary frame ownership/completion order.
            endpoint, midpoint, initial = render(3), render(2), render(1)
            np.testing.assert_allclose(endpoint, initial, rtol=3e-6, atol=2e-6)
            self.assertGreater(float(np.max(np.abs(midpoint-initial))), 1e-4)
            self.assertEqual(len(renderer.static_kernel_args), 33)
            del renderer
            hybrid.cfg.DISK_LOOP_PERIOD = 0.
            renderer = hybrid.GpuRenderer(0, 20., self.output)
            self.assertEqual(len(renderer.static_kernel_args), 29)
            render(1)


if __name__ == '__main__':
    unittest.main()
