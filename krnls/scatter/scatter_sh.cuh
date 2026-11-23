// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#ifndef DISK_COEFFICIENTS
#error DISK_SCATTER requires a coefficient material
#endif
#define SH_COUNT 6
#define SH_GROUPS ((SH_COUNT + 3) / 4)

__device__ __forceinline__ void scatter_basis(float3 v, float *y)
{
    // The monopole followed by the five real quadrupole harmonics (m = -2..2).
    y[0] = 0.282094792f;
    y[1] = 1.092548431f * v.x * v.y;
    y[2] = -1.092548431f * v.y * v.z;
    y[3] = 0.315391565f * (3.0f * v.z * v.z - 1.0f);
    y[4] = -1.092548431f * v.x * v.z;
    y[5] = 0.546274215f * (v.x * v.x - v.y * v.y);
}

// Signed azimuth velocity of the disk in the local static tetrad.
__device__ __forceinline__ float scatter_beta(float3 pos, float rho, float td, float pd)
{
    float u = 0.5f / length(pos), a = 1.0f + u;
    return -rho * pd / td * (a*a*a) / (1.0f - u);
}

// Boost a future photon direction along the phi axis. Reversing beta inverts it.
__device__ __forceinline__ float3 scatter_boost_dir(float3 d, float beta)
{
    float inv = 1.0f / (1.0f - beta * d.y);
    float transverse = sqrtf(1.0f - beta * beta) * inv;
    return make_float3(d.x * transverse, (d.y - beta) * inv, d.z * transverse);
}

__device__ __forceinline__ float3 scatter_fluid_direction(float3 pos, float3 reverse_dir, float rho, float td, float pd)
{
    float3 local = make_float3(-(pos.x*reverse_dir.x + pos.y*reverse_dir.y) / rho,
                               (pos.y*reverse_dir.x - pos.x*reverse_dir.y) / rho, -reverse_dir.z);
    return scatter_boost_dir(local, scatter_beta(pos, rho, td, pd));
}

__device__ __forceinline__ float scatter_dot4(float4 a, const float *b)
{
    return a.x*b[0] + a.y*b[1] + a.z*b[2] + a.w*b[3];
}

__device__ __forceinline__ float3 scatter_lookup(const cudaTextureObject_t *tables, float3 uvw,
                                                 float3 direction)
{
    float y[4 * SH_GROUPS] = {};
    scatter_basis(direction, y);
    // The normalized unpolarized Thomson kernel is (1 + P2(mu)/2) / (4*pi).
    // Its convolution eigenvalues are 1 for l=0 and 0.1 for l=2.
    #pragma unroll
    for (int k = 1; k < SH_COUNT; ++k) y[k] *= 0.1f;
    float3 out = make_float3(0, 0, 0);
    for (int k = 0; k < SH_GROUPS; ++k) {
        const float *b = &y[k*4];
        out.x += scatter_dot4(tex3D<float4>(tables[3*k],   uvw.x, uvw.y, uvw.z), b);
        out.y += scatter_dot4(tex3D<float4>(tables[3*k+1], uvw.x, uvw.y, uvw.z), b);
        out.z += scatter_dot4(tex3D<float4>(tables[3*k+2], uvw.x, uvw.y, uvw.z), b);
    }
    return make_float3(fmaxf(out.x, 0.0f), fmaxf(out.y, 0.0f), fmaxf(out.z, 0.0f));
}

__device__ __forceinline__ void scatter_add(float4 &emission, const cudaTextureObject_t *tables, float3 pos,
                                            float3 reverse_step, float rho, float azimuth, float td, float pd,
                                            float extinction, float g4, float albedo)
{
    if (!tables || extinction <= 0.0f || albedo == 0.0f) return;
    float3 d = scatter_fluid_direction(pos, normalize(reverse_step), rho, td, pd);
    float3 uvw = make_float3(azimuth * 0.159154943f, pos.z / (2.0f * DISK_HALF_EXTENT) + 0.5f,
                             (rho - 4.9495f) / 30.0505f);
    float3 light = scatter_lookup(tables, uvw, d);
    float strength = extinction * albedo * g4;
    emission.x = fmaf(light.x, strength, emission.x);
    emission.y = fmaf(light.y, strength, emission.y);
    emission.z = fmaf(light.z, strength, emission.z);
}
