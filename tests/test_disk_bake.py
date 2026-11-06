# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""GPU regressions for independent cloud color, density, and brightness."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import cupy as cp
    from tools import prebake_disk_noise_npgs as bake
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA GPU are required')
class CloudBakeTests(unittest.TestCase):
    def render_bake(self, **settings):
        # A synthetic constant LUT makes color changes caused by cloud masks
        # observable without relying on any ignored local cache files.
        physics = np.empty((4, 16, 4), dtype=np.float32)
        physics[...] = (0.2, 4500.0, 1.0, 0.0)
        settings = dict(R_SAMPLES=24, Z_SAMPLES=16, PHI_SAMPLES=64,
                        CLOUD_STRUCTURE=True, **settings)
        with tempfile.TemporaryDirectory() as directory, \
                patch.multiple(bake, **settings), \
                patch.object(bake.np, 'load', return_value=physics), \
                contextlib.redirect_stdout(io.StringIO()):
            return bake.main(Path(directory) / 'cloud.npy')

    def test_opacity_mask_does_not_recolor_clouds(self):
        original = self.render_bake()
        more_cores = self.render_bake(COLD_CORE_START=0.25, COLD_CORE_END=0.5)
        self.assertGreater(float(np.abs(original[..., 3] - more_cores[..., 3]).mean()), 0.01)
        np.testing.assert_array_equal(original[..., 1], more_cores[..., 1])
        # The original LUT temperature receives only a small noise perturbation.
        self.assertGreater(float(original[..., 1].min()), 4000.0)
        self.assertLess(float(original[..., 1].max()), 5000.0)

    def test_brightness_does_not_change_density_or_color(self):
        dim = self.render_bake(CLOUD_EMISSION_GAIN=1.0)
        bright = self.render_bake(CLOUD_EMISSION_GAIN=4.0)
        np.testing.assert_array_equal(dim[..., [0, 1, 3]], bright[..., [0, 1, 3]])
        np.testing.assert_allclose(bright[..., 2], 4.0 * dim[..., 2], rtol=1e-6)
        self.assertTrue(np.isfinite(bright).all())
        self.assertGreaterEqual(float(bright.min()), 0.0)
        self.assertLess(float(bright.max()), 65504.0)

    def test_thin_links_fill_gaps_without_recoloring_or_adding_cores(self):
        isolated = self.render_bake(CLOUD_LINK_DENSITY=0.0)
        linked = self.render_bake(CLOUD_LINK_DENSITY=0.10)
        np.testing.assert_array_equal(isolated[..., 1:], linked[..., 1:])
        self.assertTrue(np.all(linked[..., 0] >= isolated[..., 0]))
        gaps = (isolated[..., 0] > 1e-5) & (isolated[..., 0] < 0.003)
        self.assertGreater(int(gaps.sum()), 100)
        self.assertGreater(float((linked[..., 0] - isolated[..., 0])[gaps].mean()), 1e-5)

    def test_porosity_opens_channels_without_recoloring_or_dimming_sources(self):
        filled = self.render_bake(CLOUD_POROSITY=0.0)
        porous = self.render_bake(CLOUD_POROSITY=0.9)
        np.testing.assert_array_equal(filled[..., 1:], porous[..., 1:])
        self.assertTrue(np.all(porous[..., 0] <= filled[..., 0]))
        # Preserve the hot inner band while reducing integrated outer columns.
        np.testing.assert_array_equal(filled[0, ..., 0], porous[0, ..., 0])
        dz = 2.0 * bake.Z_MAX / filled.shape[1]
        full_tau = (filled[..., 0] * (0.6 + 8.4 * filled[..., 3])).sum(axis=1) * dz
        open_tau = (porous[..., 0] * (0.6 + 8.4 * porous[..., 3])).sum(axis=1) * dz
        self.assertLess(float(open_tau[8:].mean()), 0.5 * float(full_tau[8:].mean()))
        self.assertGreater(float(open_tau[8:].max()), 1.0)  # some thick lobes survive


if __name__ == '__main__':
    unittest.main()
