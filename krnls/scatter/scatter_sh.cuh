// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#ifndef DISK_COEFFICIENTS
#error DISK_SCATTER requires a coefficient material
#endif
#if SCATTER_ORDER < 0 || SCATTER_ORDER > 8
#error Supported SH degrees are 0 through 8
#endif
#define SH_COUNT ((SCATTER_ORDER + 1) * (SCATTER_ORDER + 1))
#define SH_GROUPS ((SH_COUNT + 3) / 4)

// Orthonormal real SH normalization, triangular index l*(l+1)/2+m.
static __device__ __constant__ float sh_norm[] = {
    2.82094792e-1f,
    4.88602512e-1f, 4.88602512e-1f,
    6.30783131e-1f, 3.64182810e-1f, 1.82091405e-1f,
    7.46352665e-1f, 3.04697200e-1f, 9.63537148e-2f, 3.93362393e-2f,
    8.46284375e-1f, 2.67618617e-1f, 6.30783131e-2f, 1.68583883e-2f, 5.96034034e-3f,
    9.35602580e-1f, 2.41571547e-1f, 4.56527313e-2f, 9.31882475e-3f, 2.19646806e-3f, 6.94584187e-4f,
    1.01710724e+0f, 2.21950995e-1f, 3.50935337e-2f, 5.84892228e-3f, 1.06786222e-3f, 2.27668991e-4f, 6.57223766e-5f,
    1.09254843e+0f, 2.06472246e-1f, 2.80973138e-2f, 3.97356023e-3f, 5.99036743e-4f, 9.98394572e-5f, 1.95801285e-5f, 5.23300945e-6f,
    1.16310662e+0f, 1.93851104e-1f, 2.31696385e-2f, 2.85198535e-3f, 3.68189726e-4f, 5.10587283e-5f, 7.87853282e-6f, 1.43841671e-6f, 3.59604179e-7f,
};

__device__ __forceinline__ void scatter_basis(float3 v, float *y)
{
    // (x+i*y)^m carries sin(theta)^m and both azimuth harmonics.
    float cm = 1.0f, sm = 0.0f, diagonal = 1.0f;
    #pragma unroll
    for (int m = 0; m <= SCATTER_ORDER; ++m) {
        float prev = 0.0f, cur = diagonal;
        #pragma unroll
        for (int l = m; l <= SCATTER_ORDER; ++l) {
            float scale = sh_norm[l * (l + 1) / 2 + m] * cur;
            y[l * (l + 1) + m] = scale * cm;
            if (m) y[l * (l + 1) - m] = scale * sm;
            float next = ((2*l + 1) * v.z * cur - (l + m) * prev) / (l - m + 1);
            prev = cur;
            cur = next;
        }
        float next_cm = cm * v.x - sm * v.y;
        sm = sm * v.x + cm * v.y;
        cm = next_cm;
        diagonal *= -(2*m + 1);
    }
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
                                                 float3 direction, float phase_g)
{
    float y[4 * SH_GROUPS] = {};
    scatter_basis(direction, y);
    // Zonal Henyey-Greenstein convolution: the degree-l eigenvalue is g^l.
    float band = 1.0f;
    for (int l = 0; l <= SCATTER_ORDER; ++l) {
        for (int i = l*l; i < (l+1)*(l+1); ++i) y[i] *= band;
        band *= phase_g;
    }
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
                                            float extinction, float g4, float phase_g, float albedo)
{
    if (!tables || extinction <= 0.0f || albedo == 0.0f) return;
    float3 d = scatter_fluid_direction(pos, normalize(reverse_step), rho, td, pd);
    float3 uvw = make_float3(azimuth * 0.159154943f, pos.z / (2.0f * DISK_HALF_EXTENT) + 0.5f,
                             (rho - 4.9495f) / 30.0505f);
    float3 light = scatter_lookup(tables, uvw, d, phase_g);
    float strength = extinction * albedo * g4;
    emission.x = fmaf(light.x, strength, emission.x);
    emission.y = fmaf(light.y, strength, emission.y);
    emission.z = fmaf(light.z, strength, emission.z);
}
