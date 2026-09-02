from __future__ import annotations

import time
import unittest

import numpy as np

import render_cpu


def synthetic_textures() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:16, 0:32].astype(np.float32)
    background = np.empty((16, 32, 4), dtype=np.float32)
    background[..., 0] = x / 31.0
    background[..., 1] = y / 15.0
    background[..., 2] = (x + 2.0 * y) / 61.0
    background[..., 3] = 1.0

    radius, height, azimuth = np.mgrid[0:12, 0:8, 0:32].astype(np.float32)
    disk = np.empty((12, 8, 32, 4), dtype=np.float32)
    disk[..., 0] = 0.08 + radius / 120.0
    disk[..., 1] = 3500.0 + radius * 350.0 + height * 20.0
    disk[..., 2] = 0.03 + azimuth / 640.0
    disk[..., 3] = 0.0

    lut_x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    color_lut = np.empty((2, 64, 4), dtype=np.float32)
    color_lut[..., 0] = lut_x
    color_lut[..., 1] = np.sqrt(lut_x)
    color_lut[..., 2] = 1.0 - 0.5 * lut_x
    color_lut[..., 3] = 1.0
    return background, disk, color_lut


def test_params(flags: render_cpu.RenderFlag, maxstep: int = 320) -> render_cpu.RenderParams:
    forward, right, up = render_cpu._camera_vectors(-10.7, -0.02, -0.4)
    return render_cpu.RenderParams(
        time=20.0,
        camera_position=(14.4, -66.0, 0.0),
        forward=forward,
        right=right,
        up=up,
        physwidth=3.2,
        physheight=2.0,
        focal_length=4.0,
        step=0.02,
        maxstep=maxstep,
        jitternum=1,
        frames=1,
        flags=flags,
    )


class Avx2RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.background, cls.disk, cls.color_lut = synthetic_textures()
        cls.native = render_cpu._native_module()

    def render_pair(self, params: render_cpu.RenderParams, width: int = 19, height: int = 11):
        native_params = params._native()
        scalar_start = time.perf_counter()
        scalar = self.native._render_new_scalar_reference(
            width, height, self.background, self.disk, self.color_lut, native_params, 13, 5, 1
        )
        scalar_seconds = time.perf_counter() - scalar_start
        avx_start = time.perf_counter()
        avx2 = self.native.render_new(
            width, height, self.background, self.disk, self.color_lut, native_params, 13, 5, 1
        )
        avx_seconds = time.perf_counter() - avx_start
        return scalar, avx2, scalar_seconds, avx_seconds

    def test_avx2_is_available(self) -> None:
        self.assertTrue(render_cpu.avx2_supported())

    def test_avx512_probe_is_safe_on_every_machine(self) -> None:
        self.assertIsInstance(render_cpu.avx512_supported(), bool)

    def test_packet_matches_scalar_with_partial_packets_and_tiles(self) -> None:
        scalar, avx2, _, _ = self.render_pair(test_params(render_cpu.RenderFlag(0), maxstep=160))
        self.assertTrue(np.isfinite(avx2).all())
        np.testing.assert_allclose(avx2, scalar, rtol=2e-4, atol=2e-5)

    def test_packet_matches_scalar_with_active_features(self) -> None:
        flags = (
            render_cpu.RenderFlag.DEPTH_JITTER
            | render_cpu.RenderFlag.RAND_SAMP_DISK
            | render_cpu.RenderFlag.OPACITY_CHANGE
            | render_cpu.RenderFlag.BACKGROUND_DOPPLER
            | render_cpu.RenderFlag.PHOTON_RING_OPTIMIZATION
        )
        scalar, avx2, _, _ = self.render_pair(test_params(flags))
        self.assertTrue(np.isfinite(avx2).all())
        np.testing.assert_allclose(avx2, scalar, rtol=3e-3, atol=2e-4)

    def test_rk4_packet_matches_scalar(self) -> None:
        flags = render_cpu.RenderFlag.USE_RK4 | render_cpu.RenderFlag.DEPTH_JITTER
        scalar, avx2, _, _ = self.render_pair(test_params(flags, maxstep=80))
        self.assertTrue(np.isfinite(avx2).all())
        np.testing.assert_allclose(avx2, scalar, rtol=5e-4, atol=5e-5)

    def test_full_rgba8_pipeline_with_multiple_workers(self) -> None:
        output = render_cpu.render_rgba8_new(
            17,
            9,
            self.background,
            self.disk,
            self.color_lut,
            test_params(render_cpu.RenderFlag.DEPTH_JITTER, maxstep=80),
            tile_size=(7, 4),
            workers=2,
        )
        self.assertEqual(output.shape, (9, 17, 4))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.all(output[..., 3] == 255))

    def test_full_avx2_math_matches_accurate_disk_path(self) -> None:
        params = render_cpu.RenderParams(
            time=20.0,
            camera_position=(0.0, -20.0, 0.0),
            forward=(0.0, 1.0, 0.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            physwidth=2.0,
            physheight=1.0,
            focal_length=4.0,
            step=0.02,
            maxstep=160,
            jitternum=1,
            frames=1,
            flags=(
                render_cpu.RenderFlag.RAND_SAMP_DISK
                | render_cpu.RenderFlag.OPACITY_CHANGE
                | render_cpu.RenderFlag.DISK_DOPPLER_FOLLOW_BACKGROUND
            ),
        )._native()
        accurate = self.native._render_new_avx2_accurate(
            67, 39, self.background, self.disk, self.color_lut, params, 32, 8, 1
        )
        full_avx2 = self.native.render_new(
            67, 39, self.background, self.disk, self.color_lut, params, 32, 8, 1
        )
        self.assertTrue(np.isfinite(full_avx2).all())
        np.testing.assert_allclose(full_avx2, accurate, rtol=2e-5, atol=2e-6)

    def test_full_avx2_math_rgba8_matches_accurate_pipeline(self) -> None:
        params = test_params(
            render_cpu.RenderFlag.DEPTH_JITTER
            | render_cpu.RenderFlag.RAND_SAMP_DISK
            | render_cpu.RenderFlag.BACKGROUND_DOPPLER,
            maxstep=180,
        )
        accurate = self.native._render_rgba8_new_avx2_accurate(
            33, 19, self.background, self.disk, self.color_lut, params._native(), 16, 7, 2, 12.0, True, False
        )
        full_avx2 = render_cpu.render_rgba8_new(
            33, 19, self.background, self.disk, self.color_lut, params, tile_size=(16, 7), workers=2
        )
        np.testing.assert_array_equal(full_avx2, accurate)


if __name__ == "__main__":
    unittest.main()
