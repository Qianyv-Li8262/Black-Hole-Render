// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Qianyv-Li8262
#pragma once

#if defined(__CUDACC__)
#define BH_TRANSFER_INLINE __device__ __forceinline__
#else
#include <math.h>
#define BH_TRANSFER_INLINE inline
#endif

namespace bh_transfer {

// Rays use p_t = -1 in isotropic Schwarzschild coordinates. With
// u = 1/(2r), |dx/dlambda| = 1/(1-u*u), and -p.u_em = td - pd*lz.
// Extinction uses the fluid-frame path length, independent of the camera.
BH_TRANSFER_INLINE float comoving_path_length(float coordinate_step, float u, float emitter_energy)
{
    return coordinate_step * (1.0f - u * u) * fabsf(emitter_energy);
}

BH_TRANSFER_INLINE float absorbed_fraction(float optical_depth)
{
    const float t = fmaxf(optical_depth, 0.0f);
    // 1-exp(-t) loses all digits for thin wisps. This expansion is accurate
    // through O(t^6) and is also used by the SIMD implementations.
    if (t < 0.01f)
        return t * (1.0f + t * (-0.5f + t * (1.0f / 6.0f + t * (-1.0f / 24.0f + t / 120.0f))));
    return 1.0f - expf(-t);
}

// A constant segment contributes T_front * (1-exp(-delta_tau)) * S,
// where S=j/sigma is radiance, not a per-length emission coefficient.
// Store total optical depth, avoiding cancellation in 1-accumulated_alpha.
BH_TRANSFER_INLINE float segment_weight(float &total_optical_depth, float segment_optical_depth)
{
    const float t = fmaxf(segment_optical_depth, 0.0f);
    const float weight = expf(-total_optical_depth) * absorbed_fraction(t);
    total_optical_depth += t;
    return weight;
}

// Exact constant-segment integral for per-length emission j, including sigma=0.
BH_TRANSFER_INLINE float emission_weight(float &tau, float extinction, float dl)
{
    float t=fmaxf(extinction*dl,0.0f);
    float phi=t<0.01f ? 1+t*(-0.5f+t*(1.0f/6+t*(-1.0f/24+t/120))) : absorbed_fraction(t)/t;
    float weight=expf(-tau)*dl*phi;
    tau+=t;
    return weight;
}

} // namespace bh_transfer

#undef BH_TRANSFER_INLINE
