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
from helpers.disk_cache import load_disk_cache

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    (ROOT / 'temp').mkdir(exist_ok=True)

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


class CacheFormatTests(unittest.TestCase):
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


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA GPU are required')
class MaterialBakeTests(unittest.TestCase):
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
