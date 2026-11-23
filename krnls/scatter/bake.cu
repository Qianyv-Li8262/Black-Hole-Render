// SPDX-License-Identifier: GPL-3.0-or-later
#include "disk.cu"
#include "scatter/bake_ray.cuh"

// GL solid-angle weights include dphi; analysis projects the incoming (future) direction.
extern "C" __global__ void projection_kernel(const float4 *directions, float *projection, int count)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= count) return;
    float4 q = directions[d];
    float y[SH_COUNT];
    scatter_basis(make_float3(q.x, q.y, q.z), y);
    for (int k = 0; k < SH_COUNT; ++k) projection[k*count+d] = y[k] * q.w;
}

extern "C" __global__ void bake_rays(float *rgb, const float4 *directions, const float2 *azimuths,
                                    int start, int cells, int count, int nr, int nz, int nphi,
                                    cudaTextureObject_t disk, cudaTextureObject_t color, float time,
                                    float step, int maxstep,
#ifdef DISK_LOOP
                                    cudaTextureObject_t disk_b, float time_a, float time_b, float weight_a,
#endif
                                    const cudaTextureObject_t *previous, float albedo,
                                    float *volume_weights, unsigned long long *truncated)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cells) return;
    int cell = start + idx;
    int ip = cell % nphi, iz = cell / nphi % nz, ir = cell / nphi / nz;
    float rho = 4.9495f + 30.0505f * ((ir + 0.5f) / nr);
    float2 cs = azimuths[ip];
    float3 pos = make_float3(rho * cs.x, rho * cs.y, DISK_HALF_EXTENT * (2.0f*(iz + 0.5f)/nz - 1.0f));
    float td, pd;
    tdpd(rho, &td, &pd);
    bake_scene scene = {};
    scene.disk = disk;
    scene.color = color;
    scene.time = time;
#ifdef DISK_LOOP
    scene.disk_b = disk_b;
    scene.time_a = time_a;
    scene.time_b = time_b;
    scene.weight_a = weight_a;
#endif
    float extinction;
    bake_sample(scene, pos, rho, atan2f(pos.y, pos.x), pd/td, 1.0f, 1.0f, extinction);
    float u = 0.5f / length(pos), a = 1.0f + u;
    float a3 = a*a*a, sigma = albedo*extinction;
    volume_weights[idx] = rho * a3*a3 * sigma*sigma;

    float n = a3 / (1.0f - u);
    float beta = scatter_beta(pos, rho, td, pd);
    // Adjacent lanes start at adjacent azimuths and march the same local direction.
    unsigned int exhausted_count = 0;
    for (int d = 0; d < count; ++d) {
        float4 q = directions[d];
        float3 dir = scatter_boost_dir(make_float3(q.x, q.y, q.z), -beta);
        float3 p = make_float3(cs.x*dir.x - cs.y*dir.y, cs.y*dir.x + cs.x*dir.y, dir.z) * -n;
        float energy = td - pd * (pos.x*p.y - pos.y*p.x);
        bool exhausted;
        float3 result = bake_integrate(pos, p, energy, scene, previous, albedo, step, maxstep, exhausted);
        exhausted_count += exhausted;
        size_t offset = ((size_t)idx * count + d) * 3;
        rgb[offset] = result.x;
        rgb[offset+1] = result.y;
        rgb[offset+2] = result.z;
    }
    if (exhausted_count) atomicAdd(truncated, (unsigned long long)exhausted_count);
}
