// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// Offline all-sky probes use disk.cu's isotropic coordinates and |p_t|=1.
// The material is stationary during a solve, so only (x,p) need advancing.
__device__ __forceinline__ void bake_rhs(float3 x, float3 p, float3 &dx, float3 &dp)
{
    float r = length(x), u = 0.5f/r, a = 1.0f+u, h = r-0.5f;
    float a2 = a*a;
    dx = p / (a2*a2);
    dp = x * (-a*(2.0f-u)/(h*h*h));
}

__device__ __forceinline__ void bake_step(float3 &x, float3 &p, float step)
{
    float r = length(x);
    float zone = 1.0f;
    if (r > 4.5f && r < 37.0f && fabsf(x.z) < DISK_HALF_EXTENT + 0.5f)
        zone = 0.05f + 0.0375f*x.z*x.z;
    float h = step * fminf(50.0f, fmaxf(0.005f, r-0.54f)) * zone;
#ifdef PHOTON_RING_OPT
    float dist = fabsf(r-1.866025f);
    h *= 0.05f + 0.95f*dist/(dist+0.12f);
#endif
    float3 k1, j1, k2, j2;
#ifdef USE_RK4
    h *= 5.0f;
#endif
    bake_rhs(x, p, k1, j1);
    bake_rhs(x+k1*(0.5f*h), p+j1*(0.5f*h), k2, j2);
#ifdef USE_RK4
    float3 k3, j3, k4, j4;
    bake_rhs(x+k2*(0.5f*h), p+j2*(0.5f*h), k3, j3);
    bake_rhs(x+k3*h, p+j3*h, k4, j4);
    x = x + (k1 + (k2+k3)*2.0f + k4)*(h/6.0f);
    p = p + (j1 + (j2+j3)*2.0f + j4)*(h/6.0f);
#else
    x = x + k2*h;
    p = p + j2*h;
#endif
    float u = 0.5f/length(x), a = 1.0f + u;
    p = normalize(p) * (a*a*a/(1.0f-u));
}

struct bake_scene {
    cudaTextureObject_t disk, color;
    float time;
#ifdef DISK_LOOP
    cudaTextureObject_t disk_b;
    float time_a, time_b, weight_a;
#endif
};

__device__ __forceinline__ float4 bake_read(cudaTextureObject_t disk, float azimuth, float rot, float v, float w)
{
    return tex3D<float4>(disk, fast_mod2pi(azimuth+rot)*0.159154943f, v, w);
}

__device__ __forceinline__ float4 bake_light(float4 material, float g, float g4, cudaTextureObject_t color)
{
#ifdef ACCRE_DISK_DOPPLER_FOLLOW_BKGD
    float4 rgb = tex2D<float4>(color, (material.y-510.0f)/20000.0f, 0.5f);
    float3 shifted = rgb_three_line_frequency_shift(make_float3(rgb.x, rgb.y, rgb.z), g) * (material.z*g);
    return make_float4(shifted.x, shifted.y, shifted.z, 1.0f);
#else
    return disk_emission_dep(fmaxf(material.y*g, 1000.0f), material.z*g4, color);
#endif
}

__device__ __forceinline__ float4 bake_sample(const bake_scene &scene, float3 pos, float rho, float azimuth,
                                              float omega, float g, float g4, float &extinction)
{
    float v = pos.z/(2.0f*DISK_HALF_EXTENT)+0.5f, w = (rho-4.9495f)/30.0505f;
#ifdef DISK_LOOP
    float4 a = bake_read(scene.disk, azimuth, omega*scene.time_a, v, w);
    float4 b = bake_read(scene.disk_b, azimuth, omega*scene.time_b, v, w);
    extinction = scene.weight_a*a.x + (1.0f-scene.weight_a)*b.x;
    return bake_light(a, g, g4, scene.color)*scene.weight_a + bake_light(b, g, g4, scene.color)*(1.0f-scene.weight_a);
#else
    float4 a = bake_read(scene.disk, azimuth, omega*scene.time, v, w);
    extinction = a.x;
    return bake_light(a, g, g4, scene.color);
#endif
}

__device__ float3 bake_integrate(float3 pos, float3 p, float receiver_energy, const bake_scene &scene,
                                 const cudaTextureObject_t *previous, float phase_g, float albedo,
                                 float step, int maxstep, bool &exhausted)
{
    float lz = pos.x*p.y - pos.y*p.x;
    float3 rgb = make_float3(0, 0, 0);
    float tau = 0.0f;
    exhausted = false;
    for (int s = 0; s < maxstep; ++s) {
        float3 start = pos;
        bake_step(pos, p, step);
        float3 mid = (pos+start)*0.5f;
        float rho2 = mid.x*mid.x + mid.y*mid.y;
        if (rho2 > 24.4974f && rho2 < 1225.0f && fabsf(mid.z) < DISK_HALF_EXTENT) {
            float rho = sqrtf(rho2), azimuth = atan2f(mid.y, mid.x);
            float td, pd;
            tdpd(rho, &td, &pd);
            float energy = td - pd*lz;
            float g = fmaxf(fabsf(receiver_energy/energy), 0.01f), g4 = g*g*g*g;
            float extinction;
            float4 emission = bake_sample(scene, mid, rho, azimuth, pd/td, g, g4, extinction);
            scatter_add(emission, previous, mid, pos-start, rho, azimuth, td, pd, extinction, g4, phase_g, albedo);
            float dl = bh_transfer::comoving_path_length(length(pos-start), 0.5f/length(mid), energy);
            float weight = bh_transfer::emission_weight(tau, extinction, dl);
            rgb = fma_3s3(make_float3(emission.x, emission.y, emission.z), weight, rgb);
        }
        float r = length(pos);
        if (!isfinite(r)) return make_float3(r, r, r);
        if (r < 0.55f || r > 140.0f) return rgb;
    }
    exhausted = true;
    return rgb;
}
