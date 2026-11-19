# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Check cache semantics and extinction-only changes through the actual baker."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from helpers.disk_cache import load_disk_cache, disk_half_extent, validate_disk_loop_pair
from tools.prebake_native34 import temperature_flare_radii

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    (ROOT / 'temp').mkdir(exist_ok=True)

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


class CacheFormatTests(unittest.TestCase):
    def test_vertical_domain_metadata_is_backward_compatible_and_validated(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            path = Path(directory) / 'disk.npy'
            self.assertEqual(disk_half_extent(path), 2.5)
            path.with_suffix('.json').write_text(json.dumps({'vertical_half_extent': 3.5}))
            self.assertEqual(disk_half_extent(path), 3.5)
            path.with_suffix('.json').write_text(json.dumps({'vertical_half_extent': -1}))
            with self.assertRaises(ValueError):
                disk_half_extent(path)

    def test_coefficient_cache_cannot_be_read_as_legacy_material(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            path = Path(directory) / 'disk.npy'
            np.save(path, np.zeros((2, 2, 2, 4), np.float32))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_disk_cache(path, coefficients=True)
            path.with_suffix('.json').write_text(json.dumps({'format': 'native34-coefficients-v1'}))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_disk_cache(path, coefficients=False)
            self.assertEqual(load_disk_cache(path, coefficients=True).dtype, np.float32)

    def test_disk_loop_pair_checks_material_inputs_and_independent_seeds(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            a, b = (Path(directory) / name for name in ('a.npy', 'b.npy'))
            settings = {'resolution': [4, 8, 16], 'temperature_noise_strength': .67}
            metadata = dict(format='native34-coefficients-v1', settings=settings,
                            physics_lut_sha256='test-physics', core_profile_sha256='test-core',
                            vertical_half_extent=4., noise_seed=1)
            a.with_suffix('.json').write_text(json.dumps(metadata))
            textures = np.zeros((4, 8, 16, 4), np.float32)
            np.save(a, textures)
            np.save(b, textures)

            def write_b(**overrides):
                b.with_suffix('.json').write_text(json.dumps({**metadata, 'noise_seed': 2, **overrides}))

            with self.assertRaisesRegex(ValueError, r'b\.npy.*coefficient metadata'):
                validate_disk_loop_pair(a, b)
            write_b(settings_sha256='different-serialization')
            self.assertEqual(validate_disk_loop_pair(a, b), 4.)

            np.save(b, textures[:, :, :8])
            with self.assertRaisesRegex(ValueError, r'b\.npy.*metadata expects'):
                validate_disk_loop_pair(a, b)
            with self.assertRaisesRegex(ValueError, r'b\.npy.*metadata expects'):
                load_disk_cache(b, coefficients=True)
            np.save(b, textures)

            for field, value in dict(settings={**settings, 'temperature_noise_strength': 0.},
                                     physics_lut_sha256='other-physics', core_profile_sha256='other-core',
                                     vertical_half_extent=2.5, noise_seed=1).items():
                with self.subTest(field=field):
                    write_b(**{field: value})
                    with self.assertRaisesRegex(ValueError, field):
                        validate_disk_loop_pair(a, b)

            for field in ('settings', 'physics_lut_sha256', 'core_profile_sha256',
                          'vertical_half_extent', 'noise_seed'):
                with self.subTest(missing=field):
                    partial = {**metadata, 'noise_seed': 2}
                    del partial[field]
                    b.with_suffix('.json').write_text(json.dumps(partial))
                    with self.assertRaisesRegex(ValueError, rf'b\.npy.*{field}'):
                        validate_disk_loop_pair(a, b)


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA GPU are required')
class MaterialBakeTests(unittest.TestCase):
    def test_outer_flare_preserves_original_inner_disk_and_extends_outer_disk(self):
        settings = json.loads((ROOT / 'cfg/disk_material.json').read_text())
        settings.update(resolution=[64, 128, 16], envelope_mode='legacy_outer_flare',
                        emission_noise_power=0, extinction_noise_power=0,
                        temperature_noise_strength=0)
        start, end = temperature_flare_radii(np.load(ROOT / 'cache/disk_lut_for_mov_disk.npy'),
                                            settings['height_temperature_start'], settings['height_temperature_end'])
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)
            def bake(name, changes):
                configuration = directory / (name + '.json')
                configuration.write_text(json.dumps({**settings, **changes}))
                output = directory / (name + '.npy')
                subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                                '--settings', str(configuration), '--output', str(output)],
                               cwd=ROOT, capture_output=True, check=True)
                return load_disk_cache(output, coefficients=True)
            extended = bake('outer', {})
            # Match physical voxel centres: both grids have dz=1/16 M.
            original = bake('original', dict(envelope_mode='legacy', vertical_half_extent=2.5,
                                            resolution=[64, 80, 16]))
            radii = 4.9495 + 30.0505*(np.arange(64)+.5)/64
            inner = extended[radii <= start, 24:104]
            original_inner = original[radii <= start]
            np.testing.assert_array_equal(inner[..., 1], original_inner[..., 1])
            # Remove per-shell noise-power normalization to isolate geometry.
            for channel in (0, 2):
                actual = inner[..., channel].astype(np.float64)
                expected = original_inner[..., channel].astype(np.float64)
                actual /= np.maximum(actual.max(axis=1, keepdims=True), 1e-30)
                expected /= np.maximum(expected.max(axis=1, keepdims=True), 1e-30)
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
            z = ((np.arange(128)+.5)/128-.5)*8
            tails = extended[radii >= end][:, np.abs(z) > 2.5, :, 2]
            self.assertGreater(float(tails.max()), 0)
            self.assertGreater(5*settings['outer_height_multiplier'], np.sqrt(27))

    def test_bounded_flare_is_thin_at_isco_and_grows_before_outer_fade(self):
        settings = json.loads((ROOT / 'cfg/disk_material.json').read_text())
        settings.update(resolution=[64, 128, 8], envelope_mode='bounded_flare',
                        height_inner=.1, height_outer=3.0,
                        height_variation=0, midplane_warp=0,
                        emission_noise_power=0, extinction_noise_power=0)
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)
            configuration = directory / 'settings.json'
            configuration.write_text(json.dumps(settings))
            output = directory / 'disk.npy'
            subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                            '--settings', str(configuration), '--output', str(output)],
                           cwd=ROOT, capture_output=True, check=True)
            data = load_disk_cache(output, coefficients=True)
            start, end = temperature_flare_radii(np.load(ROOT / 'cache/disk_lut_for_mov_disk.npy'),
                                                  settings['height_temperature_start'], settings['height_temperature_end'])
            settings['height_transition_start'], settings['height_transition_end'] = start, end
            r = 4.9495 + 30.0505 * (np.arange(64) + .5) / 64
            z = ((np.arange(128) + .5) / 128 - .5) * 2*settings["vertical_half_extent"]
            t = np.clip((r-settings['height_transition_start'])/(settings['height_transition_end']-settings['height_transition_start']), 0, 1)
            h = settings['height_inner'] + (settings['height_outer']-settings['height_inner']) * (
                t*t*(3-2*t)) ** settings['height_power']
            q = z[None, :] / h[:, None]
            edge = np.clip((1-np.abs(q))/settings['height_edge_softness'], 0, 1)
            expected = np.exp(-.25*q*q) * edge*edge*(3-2*edge)
            emission = data[:, :, 0, 2].astype(np.float64)
            # Compare the vertical profile independently of the radial physics.
            luminous = emission[:, 64] > 0
            self.assertGreater(np.count_nonzero(luminous), 20)
            np.testing.assert_allclose(emission[luminous] / emission[luminous, 64, None],
                                       expected[luminous] / expected[luminous, 64, None], rtol=2e-4, atol=2e-6)
            outside = np.broadcast_to(np.abs(q) >= 1, data.shape[:2])
            self.assertTrue((data[..., 0][outside] == 0).all())
            self.assertTrue((data[..., 2][outside] == 0).all())
            self.assertLess(2*h[0], .21)
            self.assertGreater(2*h[r >= 26][0], np.sqrt(27))
            np.testing.assert_array_equal(h[r <= start], settings["height_inner"])
            self.assertEqual(disk_half_extent(output), settings["vertical_half_extent"])
            self.assertTrue((np.diff(h) >= 0).all())

    def test_seed_is_reproducible_and_produces_independent_clouds(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)
            def bake(name, seed):
                output = directory / (name + '.npy')
                subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                    '--resolution', '16', '8', '32', '--seed', str(seed), '--output', str(output)],
                    cwd=ROOT, capture_output=True, check=True)
                self.assertEqual(json.loads(output.with_suffix('.json').read_text())['noise_seed'], seed)
                return load_disk_cache(output, coefficients=True)
            a = bake('a', 1919810)
            np.testing.assert_array_equal(a, bake('repeat', 1919810))
            b = bake('b', 2718281)
            for channel in range(3):
                self.assertFalse(np.array_equal(a[..., channel], b[..., channel]))

    def test_filtered_noise_preserves_radial_profiles_and_reduces_fine_grain(self):
        settings = json.loads((ROOT / 'cfg/disk_material.json').read_text())
        settings['resolution'] = [64, 32, 128]
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)

            def bake(name, changes):
                configuration = directory / (name + '.json')
                configuration.write_text(json.dumps({**settings, **changes}))
                output = directory / (name + '.npy')
                subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                    '--settings', str(configuration), '--output', str(output)],
                    cwd=ROOT, capture_output=True, check=True)
                return load_disk_cache(output, coefficients=True)

            legacy = bake('legacy', {'noise_mode': 'legacy'})
            filtered = bake('filtered', {'noise_mode': 'filtered',
                                         'noise_octave_start': 1, 'noise_roughness': .65})
            mixed = bake('mixed', {'noise_mode': 'filtered', 'noise_octave_start': 1,
                                   'noise_roughness': .65, 'noise_filter_mix': .35})
            disabled = bake('disabled', {'noise_mode': 'filtered', 'noise_octave_start': 1,
                                         'noise_roughness': .65, 'noise_filter_mix': 0.0})
            np.testing.assert_array_equal(disabled, legacy)
            np.testing.assert_allclose(mixed, .65*legacy+.35*filtered, rtol=3e-6, atol=1e-6)
            self.assertTrue(np.isfinite(filtered).all())
            self.assertTrue((filtered >= 0).all())
            # Extinction, temperature, and emission all retain their original
            # per-shell mean. Detail changes are not a global exposure tweak.
            np.testing.assert_allclose(filtered[..., :3].sum(axis=(1, 2), dtype=np.float64),
                                       legacy[..., :3].sum(axis=(1, 2), dtype=np.float64),
                                       rtol=3e-6, atol=1e-7)

            def azimuth_roughness(data):
                emission = data[:, 16, :, 2].astype(np.float64)
                mean = emission.mean(axis=1)
                emission = emission[mean > 1e-6] / mean[mean > 1e-6, None]
                return np.mean(np.abs(np.diff(emission, axis=1)))

            self.assertLess(azimuth_roughness(filtered), .7 * azimuth_roughness(legacy))

    def test_noise_powers_change_only_their_own_channel(self):
        settings = json.loads((ROOT / 'cfg/disk_material.json').read_text())
        settings['resolution'] = [32, 8, 16]
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)

            def bake(name, emission, extinction):
                effective = {**settings, 'emission_noise_power': emission,
                             'extinction_noise_power': extinction}
                configuration = directory / (name + '.json')
                configuration.write_text(json.dumps(effective))
                output = directory / (name + '.npy')
                subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                    '--settings', str(configuration), '--output', str(output)],
                    cwd=ROOT, capture_output=True, check=True)
                return load_disk_cache(output, coefficients=True)

            baseline = bake('baseline', 1, 1)
            emission = bake('emission', 1.8, 1)
            extinction = bake('extinction', 1, .4)
            flat = bake('flat', 0, 0)
            np.testing.assert_array_equal(baseline[..., :2], emission[..., :2])
            np.testing.assert_array_equal(baseline[..., 1:], extinction[..., 1:])
            np.testing.assert_array_equal(baseline[..., 1], flat[..., 1])
            self.assertGreater(float(np.max(np.abs(baseline[..., 2] - emission[..., 2]))), 1)
            self.assertGreater(float(np.max(np.abs(baseline[..., 0] - extinction[..., 0]))), .01)
            # Powers change local contrast while preserving each shell's
            # integrated emission/extinction, avoiding a global exposure change.
            for channel, changed, power in [(2, emission, 1.8), (0, extinction, .4)]:
                base = baseline[..., channel].astype(np.float64)
                envelope = flat[..., channel].astype(np.float64)
                q = np.divide(base, envelope, out=np.zeros_like(base), where=envelope > 0)
                expected = envelope * q ** power
                totals = expected.sum(axis=(1, 2))
                scale = np.divide(base.sum(axis=(1, 2)), totals,
                                  out=np.zeros_like(totals), where=totals > 0)
                expected *= scale[:, None, None]
                np.testing.assert_allclose(changed[..., channel], expected, rtol=5e-6, atol=1e-6)
                np.testing.assert_allclose(changed[..., channel].sum(axis=(1, 2), dtype=np.float64),
                                           base.sum(axis=(1, 2)), rtol=3e-6)

    def test_opacity_curve_changes_only_extinction_and_preserves_temperature_noise(self):
        settings = json.loads((ROOT / 'cfg/disk_material.json').read_text())
        settings['resolution'] = [32, 8, 16]
        with tempfile.TemporaryDirectory(dir=ROOT / 'temp') as directory:
            directory = Path(directory)

            def bake(name, changes):
                configuration = directory / (name + '.json')
                configuration.write_text(json.dumps({**settings, **changes}))
                output = directory / (name + '.npy')
                subprocess.run([sys.executable, str(ROOT / 'tools/prebake_native34.py'),
                    '--settings', str(configuration), '--output', str(output)],
                    cwd=ROOT, capture_output=True, check=True)
                return load_disk_cache(output, coefficients=True)

            curve = bake('curve', {})
            uniform = bake('uniform', {'opacity_inner_multiplier': .25, 'opacity_outer_multiplier': .25})
            thermal_off = bake('thermal_off', {'temperature_noise_strength': 0})
            np.testing.assert_array_equal(curve[..., 1:], uniform[..., 1:])
            np.testing.assert_array_equal(curve[..., 0], thermal_off[..., 0])
            np.testing.assert_array_equal(curve[..., 2:], thermal_off[..., 2:])
            self.assertGreater(float(np.max(np.abs(curve[..., 1] - thermal_off[..., 1]))), 1)

            r = 4.9495 + 30.0505 * (np.arange(32) + .5) / 32
            t = np.clip((r - settings['opacity_transition_start']) /
                        (settings['opacity_transition_end'] - settings['opacity_transition_start']), 0, 1)
            m = settings['opacity_inner_multiplier'] + (settings['opacity_outer_multiplier'] -
                settings['opacity_inner_multiplier']) * t*t*(3-2*t)
            expected_sigma = uniform[..., 0] * (m / .25)[:, None, None]
            np.testing.assert_allclose(curve[..., 0], expected_sigma, rtol=3e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
