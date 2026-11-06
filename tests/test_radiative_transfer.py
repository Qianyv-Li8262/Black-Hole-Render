# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Analytic transfer checks and full-renderer grey-opacity regressions."""
import math
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

import render_cpu

try:
    import cupy as cp
    from helpers.cuda_tex import (create_texture_array_2d, create_texture_array_3d,
                                  create_texture_surface_union_2d)
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


def beam_params(velocity=0.0):
    # A negligible aperture keeps the same geometric ray under a boost along
    # the viewing axis. A grey absorber's transmission must remain unchanged.
    return render_cpu.RenderParams(
        time=0.0, camera_position=(12.0, 0.0, 12.0),
        forward=(0.0, 0.0, -1.0), right=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
        velocity=(velocity, 0.0, 0.0), physwidth=1e-8, physheight=1e-8,
        focal_length=2.0, step=0.01, maxstep=8000, jitternum=1,
        flags=render_cpu.RenderFlag.OPACITY_CHANGE,
    )


def beam_textures():
    background = np.ones((4, 8, 4), dtype=np.float32)
    disk = np.zeros((8, 8, 16, 4), dtype=np.float32)
    disk[...] = (0.04, 4500.0, 0.0, 1.0)
    color = np.ones((2, 8, 4), dtype=np.float32)
    return background, disk, color


class CpuTransferTests(unittest.TestCase):
    def test_independent_coefficients_emit_with_zero_extinction(self):
        native = render_cpu._native_module()
        background, disk, color = beam_textures()
        background.fill(0)
        disk[..., 0] = 0
        disk[..., 2] = 0.04
        params = replace(beam_params(), flags=render_cpu.RenderFlag.DISK_COEFFICIENTS)
        backends = (native._render_new_scalar_reference, native._render_new_avx2_accurate,
                    native._render_new_avx2_full_math, native.render_new)
        reference = backends[0](3, 3, background, disk, color, params._native(), 3, 3, 1)
        self.assertGreater(float(reference[..., :3].min()), 0)
        for backend in backends:
            for legacy_flag in (0, render_cpu.RenderFlag.OPACITY_CHANGE):
                with self.subTest(backend=backend.__name__, legacy_flag=legacy_flag):
                    params = replace(params, flags=render_cpu.RenderFlag.DISK_COEFFICIENTS | legacy_flag)
                    actual = backend(3, 3, background, disk, color, params._native(), 3, 3, 1)
                    np.testing.assert_allclose(actual[..., :3], reference[..., :3], rtol=5e-4, atol=2e-6)

    def test_grey_extinction_is_independent_of_observer_velocity(self):
        native = render_cpu._native_module()
        textures = beam_textures()
        for backend in (native._render_new_scalar_reference,
                        native._render_new_avx2_accurate, native.render_new):
            with self.subTest(backend=backend.__name__):
                at_rest = backend(3, 3, *textures, beam_params()._native(), 3, 3, 1)
                moving = backend(3, 3, *textures, beam_params(0.6)._native(), 3, 3, 1)
                self.assertGreater(float(at_rest[..., :3].min()), 0.01)
                self.assertLess(float(at_rest[..., :3].max()), 0.8)
                np.testing.assert_allclose(moving[..., :3], at_rest[..., :3], rtol=5e-5, atol=1e-6)


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA GPU are required')
class CudaTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        # Include the production renderer and its transfer helper. These tests
        # exercise the actual compiled blend, not a Python reimplementation.
        cls.module = cp.RawModule(code=r'''
            #include "disk.cu"
            extern "C" __global__ void slab(float *out, float tau, float source,
                                           float background, int segments) {
                float total_tau = 0.0f, radiance = 0.0f;
                for (int i = 0; i < segments; ++i) {
                    float w = bh_transfer::segment_weight(total_tau, tau / segments);
                    radiance += w * source;
                }
                out[0] = radiance + background * expf(-total_tau);
                out[1] = expf(-total_tau);
            }
            extern "C" __global__ void layers(float *out, const float *data, int count) {
                float tau = 0.0f, radiance = 0.0f;
                for (int i = 0; i < count; ++i) {
                    float w = bh_transfer::segment_weight(tau, data[2*i]);
                    radiance += w * data[2*i + 1];
                }
                out[0] = radiance;
                out[1] = expf(-tau);
            }
            extern "C" __global__ void static_path(float *out, float r, float ds) {
                float u = 0.5f / r;
                float lapse = (1.0f - u) / (1.0f + u);
                out[0] = bh_transfer::comoving_path_length(ds, u, 1.0f / lapse);
            }
            extern "C" __global__ void coefficient_slab(float *out, float sigma, int segments) {
                float tau = 0, radiance = 0;
                for (int i = 0; i < segments; ++i)
                    radiance += 2.3f * bh_transfer::emission_weight(tau, sigma, 3.0f / segments);
                out[0] = radiance + 0.7f * expf(-tau);
                out[1] = expf(-tau);
            }
        ''', options=('-use_fast_math', f'-I{root / "krnls"}', '-DOPACITY_CHANGE',
                      '-DNO_DEPTH_JITTER', '-DNO_BKGD_DOPPLER'))
        cls.module.compile()
        cls.coefficient_module = cp.RawModule(
            code='#include "disk.cu"',
            options=('-use_fast_math', f'-I{root / "krnls"}', '-DDISK_COEFFICIENTS',
                     '-DOPACITY_CHANGE', '-DNO_DEPTH_JITTER', '-DNO_BKGD_DOPPLER'))
        cls.coefficient_module.compile()

    def test_independent_emission_matches_analytic_slab(self):
        output = cp.empty(2, dtype=cp.float32)
        for sigma in (0, 1e-12, 1e-4, .5, 20):
            transmission = math.exp(-3 * sigma)
            expected = 2.3 * (-math.expm1(-3 * sigma) / sigma if sigma else 3) + .7 * transmission
            for segments in (1, 128, 4096):
                with self.subTest(sigma=sigma, segments=segments):
                    self.module.get_function('coefficient_slab')((1,), (1,),
                        (output, np.float32(sigma), np.int32(segments)))
                    np.testing.assert_allclose(output.get(), (expected, transmission), rtol=3e-4, atol=2e-6)

    def slab(self, tau, source, background, segments):
        output = cp.empty(2, dtype=cp.float32)
        self.module.get_function('slab')((1,), (1,),
            (output, np.float32(tau), np.float32(source), np.float32(background), np.int32(segments)))
        return output.get()

    def test_homogeneous_medium_matches_analytic_solution(self):
        # Covers vacuum, thin mist, an emitting absorber, and an opaque cloud.
        for tau in (0.0, 1e-4, 0.02, 2.0, 20.0):
            expected_t = math.exp(-tau)
            expected_l = 2.3 * -math.expm1(-tau) + 0.7 * expected_t
            for segments in (1, 4, 128, 4096):
                with self.subTest(tau=tau, segments=segments):
                    np.testing.assert_allclose(self.slab(tau, 2.3, 0.7, segments),
                                               (expected_l, expected_t), rtol=3e-4, atol=2e-6)

    def test_sub_float_epsilon_mist_still_emits(self):
        # Each segment has tau~2.4e-8: 1-expf(-tau) used to round to zero.
        result = self.slab(1e-4, 3.0, 0.0, 4096)
        self.assertGreater(float(result[0]), 0.0)
        self.assertAlmostEqual(float(result[0]) / (3.0 * -math.expm1(-1e-4)), 1.0, places=4)

    def test_front_absorber_attenuates_rear_emitter_once(self):
        output = cp.empty(2, dtype=cp.float32)
        data = cp.asarray([2.0, 0.0, 0.4, 5.0], dtype=cp.float32)
        self.module.get_function('layers')((1,), (1,), (output, data, np.int32(2)))
        np.testing.assert_allclose(output.get(),
            (math.exp(-2.0) * 5.0 * -math.expm1(-0.4), math.exp(-2.4)), rtol=2e-6)

    def test_static_fluid_uses_proper_spatial_length(self):
        output = cp.empty(1, dtype=cp.float32)
        for radius in (5.0, 12.0, 35.0):
            self.module.get_function('static_path')((1,), (1,),
                (output, np.float32(radius), np.float32(0.2)))
            expected = 0.2 * (1.0 + 0.5 / radius) ** 2
            self.assertAlmostEqual(float(output.get()[0]), expected, places=6)

    def render_beam(self, velocity=0.0, coefficients=False, extinction=.04):
        background, disk, color = beam_textures()
        if coefficients:
            disk[..., 0] = extinction
            disk[..., 2] = .04
        background_tex = create_texture_array_2d(cp.asarray(background), 4, (1, 1, 1, 1))
        disk_tex = create_texture_array_3d(cp.asarray(disk), 4, (0, 0, 1, 1, 1))
        color_tex = create_texture_array_2d(cp.asarray(color), 4, (1, 1, 1, 1))
        output_tex, output_surface = create_texture_surface_union_2d(
            cp.zeros((3, 3, 4), dtype=cp.float32), 4, (1, 1, 0, 1))
        params = beam_params(velocity)
        args = [np.uint64(output_surface.ptr), np.uint64(background_tex.ptr),
                np.uint64(disk_tex.ptr), np.uint64(color_tex.ptr)]
        args += [np.float32(params.time)]
        args += [np.float32(x) for v in (params.camera_position, params.forward,
                 params.right, params.up, params.velocity) for x in v]
        args += [np.int32(3), np.int32(3)]
        args += [np.float32(x) for x in (params.physwidth, params.physheight,
                                       params.focal_length, params.step)]
        args += [np.int32(params.maxstep), np.int32(1), np.int32(1)]
        module = self.coefficient_module if coefficients else self.module
        module.get_function('blackholekernel')((1, 1), (3, 3), tuple(args))
        cp.cuda.Stream.null.synchronize()
        output = np.empty((3, 3, 4), dtype=np.float32)
        output_tex.ResDesc.cuArr.copy_to(output.reshape(3, 12))
        return output

    def test_full_cuda_coefficients_match_cpu_with_and_without_absorption(self):
        native = render_cpu._native_module()
        for extinction in (0, .04, 3):
            gpu = self.render_beam(coefficients=True, extinction=extinction)
            background, disk, color = beam_textures()
            disk[..., 0] = extinction
            disk[..., 2] = .04
            params = replace(beam_params(), flags=render_cpu.RenderFlag.DISK_COEFFICIENTS | render_cpu.RenderFlag.OPACITY_CHANGE)
            for backend in (native._render_new_scalar_reference, native._render_new_avx2_accurate,
                            native._render_new_avx2_full_math, native.render_new):
                with self.subTest(extinction=extinction, backend=backend.__name__):
                    cpu = backend(3, 3, background, disk, color, params._native(), 3, 3, 1)
                    np.testing.assert_allclose(gpu[..., :3], cpu[..., :3], rtol=6e-4, atol=2e-6)

    def test_full_cuda_extinction_is_independent_of_observer_velocity(self):
        at_rest = self.render_beam()
        moving = self.render_beam(0.6)
        self.assertGreater(float(at_rest[..., :3].min()), 0.01)
        self.assertLess(float(at_rest[..., :3].max()), 0.8)
        np.testing.assert_allclose(moving[..., :3], at_rest[..., :3], rtol=5e-5, atol=1e-6)

    def test_full_cuda_matches_cpu_absorption(self):
        gpu = self.render_beam()
        cpu = render_cpu._native_module().render_new(
            3, 3, *beam_textures(), beam_params()._native(), 3, 3, 1)
        np.testing.assert_allclose(gpu[..., :3], cpu[..., :3], rtol=5e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
