# SPDX-License-Identifier: GPL-3.0-or-later
"""CUDA checks for the scatter SH basis, phase convolution and table packing."""
from pathlib import Path
import unittest

import numpy as np

from bake_scatter import quadrature
from scatter_table import ScatterTable, pack_sh, table_shape

ROOT = Path(__file__).resolve().parents[1]
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False

COUNT = 6
SOURCE = r'''
#include "cuda_vec_math_utils.cuh"
#include "scatter/scatter_sh.cuh"

extern "C" __global__ void basis(const float4 *dirs, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float y[4 * SH_GROUPS] = {};
    float4 q = dirs[i];
    scatter_basis(make_float3(q.x, q.y, q.z), y);
    for (int k = 0; k < SH_COUNT; ++k) out[i * SH_COUNT + k] = y[k];
}

extern "C" __global__ void lookup(const cudaTextureObject_t *tables, const float4 *dirs,
                                  float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 q = dirs[i];
    float3 c = scatter_lookup(tables, make_float3(0.3f, 0.5f, 0.5f), make_float3(q.x, q.y, q.z));
    out[3*i] = c.x;
    out[3*i + 1] = c.y;
    out[3*i + 2] = c.z;
}
'''


@unittest.skipUnless(GPU_AVAILABLE, 'CuPy and a CUDA GPU are required')
class ScatterHarmonicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = ('-DDISK_COEFFICIENTS', '-DDISK_HALF_EXTENT=4.0f',
                   f'-I{ROOT / "krnls"}')
        module = cp.RawModule(code=SOURCE, options=options)
        cls.basis_kernel = module.get_function('basis')

    def basis(self, dirs):
        dirs = cp.asarray(np.ascontiguousarray(dirs, np.float32))
        out = cp.empty((len(dirs), COUNT), cp.float32)
        self.basis_kernel(((len(dirs) + 127)//128,), (128,), (dirs, out, np.int32(len(dirs))))
        return out.get().astype(np.float64)

    def test_basis_is_orthonormal_under_the_bake_quadrature(self):
        q = quadrature(256, 2)
        y = self.basis(q)
        gram = y.T @ (y * q[:, 3:4].astype(np.float64))
        np.testing.assert_allclose(gram, np.eye(COUNT), atol=2e-5)

    def test_lookup_matches_angular_thomson_integral_for_each_rgb_channel(self):
        q = quadrature(256, 2)
        x, y, z, weight = q.astype(np.float64).T
        # Different l=0/1/2 radiance per channel also checks the RGB plane layout.
        radiance = np.stack([np.ones_like(z), 1 + 0.6*z, 1 + 0.9*x*y + 0.3*x*z], axis=1)
        coeff = self.basis(q).T @ (radiance * weight[:, None])
        grid = [2, 2, 2]
        shape = table_shape(grid)
        cells = int(np.prod(grid))
        planes = pack_sh(np.repeat(coeff[None].astype(np.float32), cells, axis=0), shape[0])
        table = ScatterTable(dict(albedo=1.0), planes.reshape(shape))
        options = (*table.options, '-DDISK_COEFFICIENTS', '-DDISK_HALF_EXTENT=4.0f', f'-I{ROOT / "krnls"}')
        lookup = cp.RawModule(code=SOURCE, options=options).get_function('lookup')

        rng = np.random.default_rng(3)
        v = rng.normal(size=(64, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        dirs = cp.asarray(np.column_stack([v, np.zeros(len(v))]).astype(np.float32))
        out = cp.empty((len(v), 3), cp.float32)
        lookup((1,), (64,), (table.handles, dirs, out, np.int32(len(v))))
        cosine = v @ q[:, :3].T
        phase = 3.0 / (16.0 * np.pi) * (1.0 + cosine**2)
        expected = phase @ (radiance * weight[:, None])
        np.testing.assert_allclose(out.get(), expected, atol=4e-4)


if __name__ == '__main__':
    unittest.main()
