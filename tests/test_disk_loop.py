# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual CUDA renderer checks, including an analytic radial light-delay oracle."""
from pathlib import Path
import unittest
import numpy as np
from helpers.camera_motion import StaticCamera

ROOT = Path(__file__).resolve().parents[1]
try:
    import cupy as cp
    from helpers.cuda_tex import (create_texture_array_2d, create_texture_array_3d,
                                  create_texture_surface_union_2d)
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


class StaticCameraTests(unittest.TestCase):
    def test_supported_observer_preserves_position_and_proper_clock(self):
        camera = StaticCamera((40., 0., 0.), (0., 0., 0.), np.eye(3), 7.)
        for _ in range(120):
            camera.advance(40 / 120 * 79 / 81)
        np.testing.assert_array_equal(camera.position, [40, 0, 0])
        np.testing.assert_array_equal(camera.velocity, [0, 0, 0])
        self.assertAlmostEqual(camera.time, 47., places=11)
        self.assertAlmostEqual(camera.proper_time, 40 * 79 / 81, places=11)
        with self.assertRaisesRegex(ValueError, 'requires'):
            StaticCamera((40., 0., 0.), (.1, 0., 0.), np.eye(3))


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and CUDA required')
class DiskLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = ('-use_fast_math', f'-I{ROOT / "krnls"}', '-DDISK_COEFFICIENTS',
                   '-DNO_DEPTH_JITTER', '-DNO_BKGD_DOPPLER')
        cls.normal = cp.RawModule(code='#include "disk.cu"', options=options)
        cls.loop = cp.RawModule(code='#include "disk.cu"', options=options + ('-DDISK_LOOP',))
        cls.normal.compile()
        cls.loop.compile()

    @staticmethod
    def material(temperature=3500., emission=.04, extinction=.02, patterned=False):
        disk = np.empty((8, 4, 64, 4), np.float32)
        disk[...] = (extinction, temperature, emission, 0)
        if patterned:
            phi = (np.arange(64) + .5) / 64 * 2 * np.pi
            disk[..., 0] *= 1 + .6 * np.cos(phi)
            disk[..., 2] *= 1 + .8 * np.sin(3 * phi)
        return disk

    def render(self, a, b=None, time=7., period=40., epoch=0., step=.005, colored=False):
        # A radial equatorial ray admits an independent analytic delay integral.
        sky = create_texture_array_2d(cp.zeros((2, 4, 4), cp.float32), 4, (1, 1, 1, 1))
        ta = create_texture_array_3d(cp.asarray(a), 4, (0, 0, 1, 1, 1))
        tb = create_texture_array_3d(cp.asarray(b), 4, (0, 0, 1, 1, 1)) if b is not None else None
        lut = np.ones((2, 256, 4), np.float32)
        if colored:
            x = np.linspace(0, 1, 256, dtype=np.float32)
            lut[..., 0], lut[..., 1], lut[..., 2] = x*x, np.sqrt(x), 1-x
        color = create_texture_array_2d(cp.asarray(lut), 4, (1, 1, 1, 1))
        out_tex, out_surface = create_texture_surface_union_2d(cp.zeros((1, 1, 4), cp.float32), 4, (1, 1, 0, 1))
        args = [np.uint64(out_surface.ptr), np.uint64(sky.ptr), np.uint64(ta.ptr), np.uint64(color.ptr)]
        args += [np.float32(v) for v in (time, 40, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)]
        args += [np.int32(1), np.int32(1)]
        args += [np.float32(v) for v in (1e-8, 1e-8, 2, step)]
        args += [np.int32(100000), np.int32(1)]
        if tb is not None:
            args += [np.uint64(tb.ptr), np.float32(period), np.float32(20), np.float32(epoch)]
        args += [np.int32(1)]
        module = self.loop if tb is not None else self.normal
        module.get_function('blackholekernel')((1, 1), (1, 1), tuple(args))
        cp.cuda.Stream.null.synchronize()
        result = np.empty((1, 1, 4), np.float32)
        out_tex.ResDesc.cuArr.copy_to(result.reshape(1, 4))
        self.assertTrue(np.isfinite(result).all())
        return result[0, 0, :3]

    def test_period_closes_with_independent_clouds_and_negative_emission_times(self):
        a = self.material(patterned=True)
        b = self.material(7000., .09, .06, patterned=True)
        for time in (-83.25, 0., 7., 27.5):
            np.testing.assert_allclose(self.render(a, b, time), self.render(a, b, time+40), rtol=3e-6, atol=2e-7)
        self.assertGreater(float(np.max(np.abs(self.render(a, b, 7)-self.render(a, b, 27)))), 1e-3)

    def test_complementary_populations_mix_emitted_rgb_before_transfer(self):
        hot = self.material(12000., .04, .02)
        cold = self.material(1500., .04, .02)
        # Equal extinction makes the transport linear in emitted RGB. This
        # identity fails if temperatures or finished images are blended instead.
        expected = self.render(hot, colored=True) + self.render(cold, colored=True)
        mixed = self.render(hot, cold, 7., colored=True) + self.render(hot, cold, 27., colored=True)
        np.testing.assert_allclose(mixed, expected, rtol=8e-6, atol=2e-7)

    def test_retarded_turnover_matches_analytic_schwarzschild_radial_ray(self):
        a = self.material(emission=.04, extinction=0)
        b = self.material(emission=0, extinction=0)
        r = np.linspace(np.sqrt(24.4974), 35., 200001)
        # Radial null dt/dr=n=(1+1/(2r))^3/(1-1/(2r)).
        def primitive(x):
            return x - 2*np.log(x) + 1/(4*x) + 4*np.log(2*x-1)
        delay = primitive(40.) - primitive(r)
        td = (2*r+1)/np.sqrt(1+4*r*r-8*r)
        g = (81/79)/td
        dl_dr = (1-(.5/r)**2)*td
        q = np.mod((7-delay)/40+.5, 1)
        wa = .5-.5*np.cos(2*np.pi*q)
        expected = np.trapezoid(.04*g**4*dl_dr*wa, r)
        simultaneous = np.trapezoid(.04*g**4*dl_dr, r) * (.5+.5*np.cos(2*np.pi*7/40))
        actual = self.render(a, b, 7., step=.001)
        np.testing.assert_allclose(actual, expected, rtol=8e-4, atol=2e-6)
        self.assertGreater(abs(float(actual[0])-simultaneous), .05)

    def test_temporal_join_has_no_jump(self):
        a = self.material(patterned=True)
        b = self.material(7000., .09, .06, patterned=True)
        coarse = np.max(np.abs(self.render(a, b, .2) - self.render(a, b, -.2)))
        fine = np.max(np.abs(self.render(a, b, .1) - self.render(a, b, -.1)))
        self.assertLess(float(fine), float(coarse)*.65 + 8e-6)
        self.assertLess(float(fine), .01)


if __name__ == '__main__':
    unittest.main()
