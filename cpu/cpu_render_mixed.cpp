// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Qianyv-Li8262
// CPU renderer scheduling, post-processing, and Python bindings.

#include "cpu_render_common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Eight independent rays are stored structure-of-arrays style.  These helpers
// deliberately use AVX2 intrinsics: the packet path does not depend on LLVM's
// loop or SLP auto-vectorizers.
struct float2x8 {
    __m256 x;
    __m256 y;
};

struct float3x8 {
    __m256 x;
    __m256 y;
    __m256 z;
};

struct float4x8 {
    __m256 x;
    __m256 y;
    __m256 z;
    __m256 w;
};

static inline __m256 v8(float value)
{
    return _mm256_set1_ps(value);
}

static inline __m256 v8_abs(__m256 value)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
}

static inline __m256 v8_select(__m256 mask, __m256 when_true, __m256 when_false)
{
    return _mm256_or_ps(_mm256_and_ps(mask, when_true), _mm256_andnot_ps(mask, when_false));
}

static inline __m256 v8_clamp(__m256 value, float low, float high)
{
    return _mm256_min_ps(_mm256_max_ps(value, v8(low)), v8(high));
}

static inline float3x8 v3_add(float3x8 a, float3x8 b)
{
    return {_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y), _mm256_add_ps(a.z, b.z)};
}

static inline float3x8 v3_sub(float3x8 a, float3x8 b)
{
    return {_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y), _mm256_sub_ps(a.z, b.z)};
}

static inline float3x8 v3_mul(float3x8 value, __m256 scalar)
{
    return {_mm256_mul_ps(value.x, scalar), _mm256_mul_ps(value.y, scalar), _mm256_mul_ps(value.z, scalar)};
}

static inline __m256 v3_dot(float3x8 a, float3x8 b)
{
    return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a.x, b.x), _mm256_mul_ps(a.y, b.y)), _mm256_mul_ps(a.z, b.z));
}

static inline __m256 v3_length(float3x8 value)
{
    return _mm256_sqrt_ps(v3_dot(value, value));
}

static inline float3x8 v3_normalize(float3x8 value)
{
    return v3_mul(value, _mm256_div_ps(v8(1.0f), v3_length(value)));
}

static inline float4x8 v4_zero()
{
    const __m256 zero = _mm256_setzero_ps();
    return {zero, zero, zero, zero};
}

static inline float4x8 v4_add(float4x8 a, float4x8 b)
{
    return {_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y), _mm256_add_ps(a.z, b.z), _mm256_add_ps(a.w, b.w)};
}

static inline float4x8 v4_mul(float4x8 value, __m256 scalar)
{
    return {_mm256_mul_ps(value.x, scalar), _mm256_mul_ps(value.y, scalar), _mm256_mul_ps(value.z, scalar),
            _mm256_mul_ps(value.w, scalar)};
}

static inline __m256 valid_lane_mask(int lane_count)
{
    return _mm256_castsi256_ps(_mm256_setr_epi32(
        lane_count > 0 ? -1 : 0, lane_count > 1 ? -1 : 0, lane_count > 2 ? -1 : 0, lane_count > 3 ? -1 : 0,
        lane_count > 4 ? -1 : 0, lane_count > 5 ? -1 : 0, lane_count > 6 ? -1 : 0, lane_count > 7 ? -1 : 0));
}

static inline bool any_lane(__m256 mask)
{
    return _mm256_movemask_ps(mask) != 0;
}

static inline void store3(float3x8 value, float *x, float *y, float *z)
{
    _mm256_store_ps(x, value.x);
    _mm256_store_ps(y, value.y);
    _mm256_store_ps(z, value.z);
}

static inline void store4(float4x8 value, float *x, float *y, float *z, float *w)
{
    _mm256_store_ps(x, value.x);
    _mm256_store_ps(y, value.y);
    _mm256_store_ps(z, value.z);
    _mm256_store_ps(w, value.w);
}

static inline float4x8 load4(const float *x, const float *y, const float *z, const float *w)
{
    return {_mm256_load_ps(x), _mm256_load_ps(y), _mm256_load_ps(z), _mm256_load_ps(w)};
}

static inline __m256i pcg_hash_v8(__m256i input)
{
    const __m256i state = _mm256_add_epi32(_mm256_mullo_epi32(input, _mm256_set1_epi32(747796405)),
                                           _mm256_set1_epi32(static_cast<int>(2891336453u)));
    const __m256i shift = _mm256_add_epi32(_mm256_srli_epi32(state, 28), _mm256_set1_epi32(4));
    const __m256i word =
        _mm256_mullo_epi32(_mm256_xor_si256(_mm256_srlv_epi32(state, shift), state), _mm256_set1_epi32(277803737));
    return _mm256_xor_si256(_mm256_srli_epi32(word, 22), word);
}

static inline __m256i reverse_bits_v8(__m256i value)
{
    value = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(value, _mm256_set1_epi32(0x55555555)), 1),
                            _mm256_and_si256(_mm256_srli_epi32(value, 1), _mm256_set1_epi32(0x55555555)));
    value = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(value, _mm256_set1_epi32(0x33333333)), 2),
                            _mm256_and_si256(_mm256_srli_epi32(value, 2), _mm256_set1_epi32(0x33333333)));
    value = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(value, _mm256_set1_epi32(0x0f0f0f0f)), 4),
                            _mm256_and_si256(_mm256_srli_epi32(value, 4), _mm256_set1_epi32(0x0f0f0f0f)));
    value = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(value, _mm256_set1_epi32(0x00ff00ff)), 8),
                            _mm256_and_si256(_mm256_srli_epi32(value, 8), _mm256_set1_epi32(0x00ff00ff)));
    return _mm256_or_si256(_mm256_slli_epi32(value, 16), _mm256_srli_epi32(value, 16));
}

static inline __m256 uint32_to_unit_float_v8(__m256i value)
{
    const __m256 high31 = _mm256_cvtepi32_ps(_mm256_srli_epi32(value, 1));
    const __m256 low_bit = _mm256_cvtepi32_ps(_mm256_and_si256(value, _mm256_set1_epi32(1)));
    return _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(high31, high31), low_bit), v8(2.3283064365386963e-10f));
}

static inline __m256 rand_float_v8(__m256i seed)
{
    seed = _mm256_xor_si256(_mm256_xor_si256(seed, _mm256_set1_epi32(61)), _mm256_srli_epi32(seed, 16));
    seed = _mm256_mullo_epi32(seed, _mm256_set1_epi32(9));
    seed = _mm256_xor_si256(seed, _mm256_srli_epi32(seed, 4));
    seed = _mm256_mullo_epi32(seed, _mm256_set1_epi32(static_cast<int>(0x27d4eb2du)));
    seed = _mm256_xor_si256(seed, _mm256_srli_epi32(seed, 15));
    return uint32_to_unit_float_v8(seed);
}

static inline float2x8 hammersley_v8(int index, int count, __m256i pixel_x, __m256i pixel_y, std::uint32_t frames)
{
    const __m256 h_x = v8(static_cast<float>(index) / static_cast<float>(count));
    const __m256 h_y = _mm256_mul_ps(uint32_to_unit_float_v8(reverse_bits_v8(_mm256_set1_epi32(index))), v8(1.0f));
    const __m256i frame_seed = _mm256_set1_epi32(static_cast<int>(frames * 1919810u));
    const __m256 shift_x = rand_float_v8(_mm256_xor_si256(_mm256_xor_si256(pixel_x, pcg_hash_v8(pixel_y)), frame_seed));
    const __m256 shift_y = rand_float_v8(_mm256_xor_si256(
        _mm256_xor_si256(pixel_y, pcg_hash_v8(_mm256_mullo_epi32(pixel_x, _mm256_set1_epi32(114514)))), frame_seed));
    const __m256 sum_x = _mm256_add_ps(h_x, shift_x);
    const __m256 sum_y = _mm256_add_ps(h_y, shift_y);
    return {_mm256_sub_ps(sum_x, _mm256_floor_ps(sum_x)), _mm256_sub_ps(sum_y, _mm256_floor_ps(sum_y))};
}

static void render_one_tile_mixed(float *raw, const Texture2D &background, const Texture3D &disk,
                                  const Texture2D &color_lut, const render_params &params, int tile_width,
                                  int tile_height, int tile_x, int tile_y)
{
    const int x_begin = tile_x * tile_width;
    const int y_begin = tile_y * tile_height;
    const int x_end = std::min(x_begin + tile_width, params.imgwidth);
    const int y_end = std::min(y_begin + tile_height, params.imgheight);

    const float3 fwd = make_float3(params.fwd_x, params.fwd_y, params.fwd_z);
    const float3 right = make_float3(params.right_x, params.right_y, params.right_z);
    const float3 up = make_float3(params.up_x, params.up_y, params.up_z);
    const float3 beta = make_float3(params.vfwd, params.vright, params.vup);
    const float gamma = rsqrtf_cpu(1.0f - beta * beta);
    const float3 initial_cam_pos = make_float3(params.cam_pos_x, params.cam_pos_y, params.cam_pos_z);
    const float camera_radius = length(initial_cam_pos);
    const float initial_u = 1.0f / (2.0f * camera_radius);
    const float initial_upl = 1.0f + initial_u;
    const float initial_umi = 1.0f - initial_u;
    const float initial_factor = initial_upl / initial_umi;
    const float3 beta_global = beta.x * fwd + beta.y * right + beta.z * up;
    const float3 e0 = beta_global * gamma / (initial_upl * initial_upl);
    float3 e1_fwd = boost(beta, make_float3(1.0f, 0.0f, 0.0f), gamma);
    e1_fwd = (e1_fwd.x * fwd + e1_fwd.y * right + e1_fwd.z * up) / (initial_upl * initial_upl);
    float3 e2_right = boost(beta, make_float3(0.0f, 1.0f, 0.0f), gamma);
    e2_right = (e2_right.x * fwd + e2_right.y * right + e2_right.z * up) / (initial_upl * initial_upl);
    float3 e3_up = boost(beta, make_float3(0.0f, 0.0f, 1.0f), gamma);
    e3_up = (e3_up.x * fwd + e3_up.y * right + e3_up.z * up) / (initial_upl * initial_upl);

    const bool use_rk4 = has_flag(params, USE_RK4);
    const bool use_depth_jitter = has_flag(params, DEPTH_JITTER);
    const bool random_disk_sample = has_flag(params, RAND_SAMP_DISK);
    const bool photon_ring_optimization = has_flag(params, PHOTON_RING_OPTIMIZATION);
    const bool opacity_change = has_flag(params, OPACITY_CHANGE);
    const bool disk_doppler_follows_background = has_flag(params, DISK_DOPPLER_FOLLOW_BACKGROUND);
    const bool background_doppler = has_flag(params, BACKGROUND_DOPPLER);

    const float3x8 e0v{v8(e0.x), v8(e0.y), v8(e0.z)};
    const float3x8 e1v{v8(e1_fwd.x), v8(e1_fwd.y), v8(e1_fwd.z)};
    const float3x8 e2v{v8(e2_right.x), v8(e2_right.y), v8(e2_right.z)};
    const float3x8 e3v{v8(e3_up.x), v8(e3_up.y), v8(e3_up.z)};
    const float3x8 initial_pos_v{v8(initial_cam_pos.x), v8(initial_cam_pos.y), v8(initial_cam_pos.z)};
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = v8(1.0f);

    alignas(32) float lane_a[8];
    alignas(32) float lane_b[8];
    alignas(32) float lane_c[8];
    alignas(32) float lane_d[8];
    alignas(32) float lane_e[8];
    alignas(32) float lane_f[8];
    alignas(32) float lane_g[8];
    alignas(32) float lane_h[8];
    alignas(32) float lane_i[8];
    alignas(32) float lane_j[8];
    alignas(32) float lane_k[8];
    alignas(32) float lane_l[8];
    alignas(32) float lane_m[8];

    for (int pixel_y = y_begin; pixel_y < y_end; ++pixel_y) {
        for (int packet_x = x_begin; packet_x < x_end; packet_x += 8) {
            const int lane_count = std::min(8, x_end - packet_x);
            const __m256 valid_mask = valid_lane_mask(lane_count);
            const __m256i pixel_x_v = _mm256_setr_epi32(packet_x, packet_x + 1, packet_x + 2, packet_x + 3,
                                                        packet_x + 4, packet_x + 5, packet_x + 6, packet_x + 7);
            const __m256i pixel_y_v = _mm256_set1_epi32(pixel_y);
            float4x8 buffer = v4_zero();

            for (int sample = 0; sample < params.jitternum; ++sample) {
                const float2x8 jitter = hammersley_v8(sample, params.jitternum, pixel_x_v, pixel_y_v, 1u);
                const __m256 physical_x =
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_div_ps(_mm256_add_ps(_mm256_cvtepi32_ps(pixel_x_v), jitter.x),
                                                              v8(static_cast<float>(params.imgwidth))),
                                                v8(0.5f)),
                                  v8(params.physwidth));
                const __m256 physical_y =
                    _mm256_mul_ps(_mm256_sub_ps(_mm256_div_ps(_mm256_add_ps(_mm256_cvtepi32_ps(pixel_y_v), jitter.y),
                                                              v8(static_cast<float>(params.imgheight))),
                                                v8(0.5f)),
                                  v8(params.physheight));

                const float3x8 camera_ray =
                    v3_normalize({v8(params.focal_length), physical_x, _mm256_sub_ps(zero, physical_y)});
                float3x8 cam_pos = initial_pos_v;
                __m256 delta_t = zero;
                __m256 radius = v8(camera_radius);
                __m256 u = v8(initial_u);
                __m256 upl = v8(initial_upl);
                __m256 umi = v8(initial_umi);
                __m256 n = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(upl, upl), upl), umi);

                float3x8 direction = v3_normalize(v3_sub(
                    v3_add(v3_add(v3_mul(e1v, camera_ray.x), v3_mul(e2v, camera_ray.y)), v3_mul(e3v, camera_ray.z)),
                    e0v));
                if (use_depth_jitter) {
                    const __m256i depth_seed = pcg_hash_v8(_mm256_xor_si256(
                        pixel_x_v,
                        pcg_hash_v8(_mm256_xor_si256(
                            pixel_y_v, pcg_hash_v8(_mm256_xor_si256(_mm256_set1_epi32(sample),
                                                                    pcg_hash_v8(_mm256_set1_epi32(params.frames))))))));
                    __m256 depth_distance = _mm256_mul_ps(uint32_to_unit_float_v8(depth_seed),
                                                          _mm256_max_ps(_mm256_sub_ps(radius, v8(1.5f)), zero));
                    depth_distance = _mm256_div_ps(depth_distance, v8(10.0f));
                    depth_distance = _mm256_mul_ps(depth_distance, v8(params.step));
                    cam_pos = v3_add(cam_pos, v3_mul(direction, depth_distance));
                }

                float3x8 p = v3_mul(direction, n);
                const float3x8 p_init = p;
                const __m256 lz = _mm256_sub_ps(_mm256_mul_ps(cam_pos.x, p.y), _mm256_mul_ps(cam_pos.y, p.x));
                __m256 active = valid_mask;
                float4x8 accumulated = v4_zero();

                for (int step_index = 0; step_index < params.maxstep && any_lane(active); ++step_index) {
                    const float3x8 prev_pos = cam_pos;
                    const __m256 prev_dt = delta_t;

                    if (use_rk4) {
                        __m256 rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        __m256 gravity = _mm256_sub_ps(
                            zero,
                            _mm256_max_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf))));
                        __m256 uplsq = _mm256_mul_ps(upl, upl);
                        __m256 uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k11 = v3_mul(p, uu);
                        const float3x8 k12 = v3_mul(cam_pos, gravity);
                        const __m256 k_t1 = _mm256_div_ps(uplsq, _mm256_mul_ps(umi, umi));
                        const __m256 in_volume =
                            _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(radius, v8(4.5f), _CMP_GT_OQ),
                                                        _mm256_cmp_ps(radius, v8(27.0f), _CMP_LT_OQ)),
                                          _mm256_cmp_ps(v8_abs(cam_pos.z), v8(3.0f), _CMP_LT_OQ));
                        __m256 disk_zone = _mm256_mul_ps(cam_pos.z, cam_pos.z);
                        disk_zone = _mm256_mul_ps(disk_zone, v8(0.25f));
                        disk_zone = _mm256_mul_ps(disk_zone, v8(0.15f));
                        disk_zone = _mm256_add_ps(v8(0.05f), disk_zone);
                        const __m256 zone_multiplier = v8_select(in_volume, disk_zone, one);
                        __m256 current_step =
                            _mm256_mul_ps(v8(params.step), v8_clamp(_mm256_sub_ps(radius, v8(0.54f)), 0.005f, 50.0f));
                        current_step = _mm256_mul_ps(current_step, zone_multiplier);
                        current_step = _mm256_mul_ps(current_step, v8(5.0f));
                        if (photon_ring_optimization) {
                            const __m256 distance = v8_abs(_mm256_sub_ps(radius, v8(1.866025f)));
                            current_step = _mm256_mul_ps(
                                current_step,
                                _mm256_add_ps(
                                    v8(0.05f),
                                    _mm256_mul_ps(v8(0.95f),
                                                  _mm256_div_ps(distance, _mm256_add_ps(distance, v8(0.12f))))));
                        }

                        __m256 step_half = _mm256_mul_ps(current_step, v8(0.5f));
                        float3x8 pos_tmp = v3_add(cam_pos, v3_mul(k11, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm256_div_ps(one, _mm256_mul_ps(v8(2.0f), radius));
                        upl = _mm256_add_ps(one, u);
                        umi = _mm256_sub_ps(one, u);
                        rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        gravity = _mm256_sub_ps(
                            zero,
                            _mm256_max_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm256_mul_ps(upl, upl);
                        uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k21 = v3_mul(v3_add(p, v3_mul(k12, step_half)), uu);
                        const float3x8 k22 = v3_mul(pos_tmp, gravity);
                        const __m256 k_t2 = _mm256_div_ps(uplsq, _mm256_mul_ps(umi, umi));

                        pos_tmp = v3_add(cam_pos, v3_mul(k21, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm256_div_ps(one, _mm256_mul_ps(v8(2.0f), radius));
                        upl = _mm256_add_ps(one, u);
                        umi = _mm256_sub_ps(one, u);
                        rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        gravity = _mm256_sub_ps(
                            zero,
                            _mm256_max_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm256_mul_ps(upl, upl);
                        uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k31 = v3_mul(v3_add(p, v3_mul(k22, step_half)), uu);
                        const float3x8 k32 = v3_mul(pos_tmp, gravity);
                        const __m256 k_t3 = _mm256_div_ps(uplsq, _mm256_mul_ps(umi, umi));

                        pos_tmp = v3_add(cam_pos, v3_mul(k31, current_step));
                        radius = v3_length(pos_tmp);
                        u = _mm256_div_ps(one, _mm256_mul_ps(v8(2.0f), radius));
                        upl = _mm256_add_ps(one, u);
                        umi = _mm256_sub_ps(one, u);
                        rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        gravity = _mm256_sub_ps(
                            zero,
                            _mm256_max_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm256_mul_ps(upl, upl);
                        uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k41 = v3_mul(v3_add(p, v3_mul(k32, current_step)), uu);
                        const float3x8 k42 = v3_mul(pos_tmp, gravity);
                        const __m256 k_t4 = _mm256_div_ps(uplsq, _mm256_mul_ps(umi, umi));

                        step_half = _mm256_mul_ps(current_step, v8(0.16666666667f));
                        float3x8 position_sum = v3_add(k11, k41);
                        position_sum = v3_add(position_sum, v3_mul(k21, v8(2.0f)));
                        position_sum = v3_add(position_sum, v3_mul(k31, v8(2.0f)));
                        cam_pos = v3_add(cam_pos, v3_mul(position_sum, step_half));
                        float3x8 momentum_sum = v3_add(k12, k42);
                        momentum_sum = v3_add(momentum_sum, v3_mul(k22, v8(2.0f)));
                        momentum_sum = v3_add(momentum_sum, v3_mul(k32, v8(2.0f)));
                        p = v3_add(p, v3_mul(momentum_sum, step_half));
                        __m256 time_sum = _mm256_add_ps(k_t1, k_t4);
                        time_sum = _mm256_add_ps(time_sum, _mm256_mul_ps(v8(2.0f), k_t2));
                        time_sum = _mm256_add_ps(time_sum, _mm256_mul_ps(v8(2.0f), k_t3));
                        delta_t = _mm256_add_ps(delta_t, _mm256_mul_ps(step_half, time_sum));
                    } else {
                        __m256 rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        __m256 gravity =
                            _mm256_sub_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf)));
                        __m256 uplsq = _mm256_mul_ps(upl, upl);
                        __m256 uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k11 = v3_mul(p, uu);
                        const float3x8 k12 = v3_mul(cam_pos, gravity);
                        const __m256 in_volume =
                            _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(radius, v8(4.5f), _CMP_GT_OQ),
                                                        _mm256_cmp_ps(radius, v8(37.0f), _CMP_LT_OQ)),
                                          _mm256_cmp_ps(v8_abs(cam_pos.z), v8(3.0f), _CMP_LT_OQ));
                        __m256 disk_zone = _mm256_mul_ps(cam_pos.z, cam_pos.z);
                        disk_zone = _mm256_mul_ps(disk_zone, v8(0.25f));
                        disk_zone = _mm256_mul_ps(disk_zone, v8(0.15f));
                        disk_zone = _mm256_add_ps(v8(0.05f), disk_zone);
                        const __m256 zone_multiplier = v8_select(in_volume, disk_zone, one);
                        __m256 current_step =
                            _mm256_mul_ps(v8(params.step), v8_clamp(_mm256_sub_ps(radius, v8(0.54f)), 0.005f, 50.0f));
                        current_step = _mm256_mul_ps(current_step, zone_multiplier);
                        if (photon_ring_optimization) {
                            const __m256 distance = v8_abs(_mm256_sub_ps(radius, v8(1.866025f)));
                            current_step = _mm256_mul_ps(
                                current_step,
                                _mm256_add_ps(
                                    v8(0.05f),
                                    _mm256_mul_ps(v8(0.95f),
                                                  _mm256_div_ps(distance, _mm256_add_ps(distance, v8(0.12f))))));
                        }
                        const __m256 step_half = _mm256_mul_ps(current_step, v8(0.5f));
                        const float3x8 pos_tmp = v3_add(cam_pos, v3_mul(k11, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm256_div_ps(one, _mm256_mul_ps(v8(2.0f), radius));
                        upl = _mm256_add_ps(one, u);
                        umi = _mm256_sub_ps(one, u);
                        rmhalf = _mm256_sub_ps(radius, v8(0.5f));
                        gravity =
                            _mm256_sub_ps(zero, _mm256_div_ps(_mm256_mul_ps(upl, _mm256_sub_ps(v8(2.0f), u)),
                                                              _mm256_mul_ps(_mm256_mul_ps(rmhalf, rmhalf), rmhalf)));
                        uplsq = _mm256_mul_ps(upl, upl);
                        uu = _mm256_div_ps(one, _mm256_mul_ps(uplsq, uplsq));
                        const float3x8 k21 = v3_mul(v3_add(p, v3_mul(k12, step_half)), uu);
                        const float3x8 k22 = v3_mul(pos_tmp, gravity);
                        const __m256 k_t2 = _mm256_div_ps(uplsq, _mm256_mul_ps(umi, umi));
                        cam_pos = v3_add(cam_pos, v3_mul(k21, current_step));
                        p = v3_add(p, v3_mul(k22, current_step));
                        delta_t = _mm256_add_ps(delta_t, _mm256_mul_ps(current_step, k_t2));
                    }

                    radius = v3_length(cam_pos);
                    u = _mm256_div_ps(one, _mm256_mul_ps(v8(2.0f), radius));
                    upl = _mm256_add_ps(one, u);
                    umi = _mm256_sub_ps(one, u);
                    n = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(upl, upl), upl), umi);
                    p = v3_mul(v3_normalize(p), n);

                    float3x8 disk_pos;
                    __m256 disk_time;
                    if (random_disk_sample) {
                        const __m256i random_seed = pcg_hash_v8(_mm256_xor_si256(
                            pixel_x_v,
                            pcg_hash_v8(_mm256_xor_si256(
                                pixel_y_v, pcg_hash_v8(_mm256_xor_si256(_mm256_set1_epi32(step_index),
                                                                        pcg_hash_v8(_mm256_set1_epi32(sample))))))));
                        const __m256 random_value = rand_float_v8(random_seed);
                        disk_pos =
                            v3_add(v3_mul(cam_pos, random_value), v3_mul(prev_pos, _mm256_sub_ps(one, random_value)));
                        disk_time = _mm256_add_ps(_mm256_mul_ps(delta_t, random_value),
                                                  _mm256_mul_ps(prev_dt, _mm256_sub_ps(one, random_value)));
                    } else {
                        disk_pos = v3_mul(v3_add(cam_pos, prev_pos), v8(0.5f));
                        disk_time = _mm256_mul_ps(_mm256_add_ps(prev_dt, delta_t), v8(0.5f));
                    }

                    const __m256 disk_radius_sq =
                        _mm256_add_ps(_mm256_mul_ps(disk_pos.x, disk_pos.x), _mm256_mul_ps(disk_pos.y, disk_pos.y));
                    __m256 in_disk =
                        _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(disk_radius_sq, v8(24.4974f), _CMP_GT_OQ),
                                                    _mm256_cmp_ps(disk_radius_sq, v8(1225.0f), _CMP_LT_OQ)),
                                      _mm256_cmp_ps(v8_abs(cam_pos.z), v8(2.5f), _CMP_LT_OQ));
                    in_disk = _mm256_and_ps(in_disk, active);

                    if (any_lane(in_disk)) {

                        store3(disk_pos, lane_a, lane_b, lane_c);
                        _mm256_store_ps(lane_d, disk_time);
                        store3(prev_pos, lane_e, lane_f, lane_g);
                        _mm256_store_ps(lane_h, radius);
                        _mm256_store_ps(lane_i, lz);
                        store3(p_init, lane_j, lane_k, lane_l);
                        _mm256_store_ps(lane_m, v3_length(v3_sub(cam_pos, prev_pos)));
                        alignas(32) float accum_x[8], accum_y[8], accum_z[8], accum_w[8];
                        store4(accumulated, accum_x, accum_y, accum_z, accum_w);
                        const int disk_bits = _mm256_movemask_ps(in_disk);

                        for (int lane = 0; lane < lane_count; ++lane) {
                            if ((disk_bits & (1 << lane)) == 0)
                                continue;
                            const float disk_radius =
                                std::sqrt(lane_a[lane] * lane_a[lane] + lane_b[lane] * lane_b[lane]);
                            float td = 0.0f;
                            float pd = 0.0f;
                            tdpd(disk_radius, &td, &pd);
                            const float rotation = pd * (params.time - lane_d[lane]) / td;
                            const float phi_final = fast_mod2pi(std::atan2(lane_b[lane], lane_a[lane]) + rotation);
                            const float4 disk_parameters =
                                disk.sample(phi_final * 0.15915494f, (lane_c[lane] / 2.5f) / 2.0f + 0.5f,
                                            (disk_radius - 4.9495f) / 30.0505f);
                            const float p_init_dot_e0 = lane_j[lane] * e0.x + lane_k[lane] * e0.y + lane_l[lane] * e0.z;
                            const float g = std::fmax(
                                (std::fabs((initial_factor * gamma + p_init_dot_e0) / (td - pd * lane_i[lane])) -
                                 1.0f) +
                                    1.0f,
                                0.01f);
                            const float prev_length =
                                std::sqrt(lane_e[lane] * lane_e[lane] + lane_f[lane] * lane_f[lane] +
                                          lane_g[lane] * lane_g[lane]);
                            const float uuu = 1.0f + 1.0f / (prev_length + lane_h[lane]);
                            const float g4 = g * g * g * g;
                            const float step_len = lane_m[lane];
                            const float kzg4 = 2.0f * disk_parameters.z;
                            float step_opacity;
                            if (opacity_change) {
                                const float temp_eff = disk_parameters.y * g;
                                const float cold_factor = 1.0f + 2.0f * saturate((5000.0f - temp_eff) / 3000.0f);
                                step_opacity = disk_parameters.x * 3.0f * uuu * uuu *
                                               std::fma(step_len, -std::exp(-kzg4 * kzg4), step_len) / g * cold_factor;
                            } else {
                                step_opacity = disk_parameters.x * 1.7f * uuu * uuu *
                                               std::fma(step_len, -std::exp(-kzg4 * kzg4), step_len) / g;
                            }
                            const float temp_exp = -std::exp(-step_opacity);
                            float4 emission;
                            if (disk_doppler_follows_background) {
                                const float4 color = color_lut.sample((disk_parameters.y - 510.0f) / 20000.0f, 0.5f);
                                const float3 shifted_rgb =
                                    rgb_three_line_frequency_shift(make_float3(color.x, color.y, color.z), g);
                                emission = make_float4(shifted_rgb.x * disk_parameters.z * g4,
                                                       shifted_rgb.y * disk_parameters.z * g4,
                                                       shifted_rgb.z * disk_parameters.z * g4, 1.0f);
                            } else {
                                emission = disk_emission_dep(std::fmax(disk_parameters.y * g, 1000.0f),
                                                             disk_parameters.z * g4, color_lut);
                            }

                            float temp_calc = std::fma(emission.x, -accum_w[lane], emission.x);
                            accum_x[lane] += std::fma(temp_calc, temp_exp, temp_calc);
                            temp_calc = std::fma(emission.y, -accum_w[lane], emission.y);
                            accum_y[lane] += std::fma(temp_calc, temp_exp, temp_calc);
                            temp_calc = std::fma(emission.z, -accum_w[lane], emission.z);
                            accum_z[lane] += std::fma(temp_calc, temp_exp, temp_calc);
                            temp_calc = 1.0f - accum_w[lane];
                            accum_w[lane] += std::fma(temp_calc, temp_exp, temp_calc);
                        }
                        accumulated = load4(accum_x, accum_y, accum_z, accum_w);
                    }

                    active = _mm256_and_ps(active,
                                           _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(radius, v8(0.55f), _CMP_GE_OQ),
                                                                       _mm256_cmp_ps(radius, v8(140.0f), _CMP_LE_OQ)),
                                                         _mm256_cmp_ps(accumulated.w, v8(0.99f), _CMP_LE_OQ)));
                }

                store3(p, lane_a, lane_b, lane_c);
                _mm256_store_ps(lane_d, radius);
                store3(p_init, lane_e, lane_f, lane_g);
                alignas(32) float accum_x[8], accum_y[8], accum_z[8], accum_w[8];
                store4(accumulated, accum_x, accum_y, accum_z, accum_w);
                alignas(32) float color_x[8] = {}, color_y[8] = {}, color_z[8] = {}, color_w[8] = {};

                for (int lane = 0; lane < lane_count; ++lane) {
                    float4 color;
                    if (lane_d[lane] >= 0.55f && !std::isnan(lane_d[lane])) {
                        const float inv_p_length =
                            1.0f / std::sqrt(lane_a[lane] * lane_a[lane] + lane_b[lane] * lane_b[lane] +
                                             lane_c[lane] * lane_c[lane]);
                        const float final_x = lane_a[lane] * inv_p_length;
                        const float final_y = lane_b[lane] * inv_p_length;
                        const float final_z = lane_c[lane] * inv_p_length;
                        const float phi = std::atan2(final_y, -final_x);
                        const float theta = std::asin(-final_z);
                        float4 bkgd = background.sample(phi * 0.1591549f + 0.5f, theta * 0.3183099f + 0.5f);
                        if (background_doppler) {
                            float g_sky = std::fabs(initial_factor * gamma + lane_e[lane] * e0.x + lane_f[lane] * e0.y +
                                                    lane_g[lane] * e0.z);
                            g_sky = std::fmin(std::fmax(g_sky, 1e-4f), 20.0f);
                            const float3 shifted_rgb =
                                rgb_three_line_frequency_shift(make_float3(bkgd.x, bkgd.y, bkgd.z), g_sky);
                            bkgd.x = shifted_rgb.x;
                            bkgd.y = shifted_rgb.y;
                            bkgd.z = shifted_rgb.z;
                        }
                        const float remaining = 1.0f - accum_w[lane];
                        color = make_float4(accum_x[lane] + bkgd.x * remaining, accum_y[lane] + bkgd.y * remaining,
                                            accum_z[lane] + bkgd.z * remaining, accum_w[lane] + bkgd.w * remaining);
                    } else {
                        color = make_float4(accum_x[lane], accum_y[lane], accum_z[lane], 1.0f);
                    }
                    color_x[lane] = color.x;
                    color_y[lane] = color.y;
                    color_z[lane] = color.z;
                    color_w[lane] = color.w;
                }
                buffer = v4_add(buffer, load4(color_x, color_y, color_z, color_w));
            }

            buffer = v4_mul(buffer, v8(1.0f / static_cast<float>(params.jitternum)));
            store4(buffer, lane_a, lane_b, lane_c, lane_d);
            for (int lane = 0; lane < lane_count; ++lane) {
                float *output = raw + (static_cast<std::size_t>(pixel_y) * params.imgwidth + packet_x + lane) * 4;
                output[0] = lane_a[lane];
                output[1] = lane_b[lane];
                output[2] = lane_c[lane];
                output[3] = 0.0f;
            }
        }
    }
}

struct FloatImage {
    int width;
    int height;
    bool linear_filter;
    std::vector<float> pixels;

    FloatImage(int image_width, int image_height, bool use_linear_filter = true)
        : width(image_width), height(image_height), linear_filter(use_linear_filter),
          pixels(static_cast<std::size_t>(image_width) * image_height * 4)
    {
    }

    Texture2D texture() const
    {
        return {pixels.data(), width, height, 4, linear_filter};
    }

    void write(int x, int y, float4 value)
    {
        float *destination = pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4;
        destination[0] = value.x;
        destination[1] = value.y;
        destination[2] = value.z;
        destination[3] = value.w;
    }
};

static int effective_worker_count(int requested_workers, int work_items)
{
    int worker_count = requested_workers;
    if (worker_count <= 0)
        worker_count = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, std::min(worker_count, work_items));
}
template <typename Function> static void parallel_rows(int height, int requested_workers, Function &&function)
{
    const int worker_count = effective_worker_count(requested_workers, height);
    if (worker_count == 1) {
        for (int y = 0; y < height; ++y)
            function(y);
        return;
    }

    std::atomic<int> next_row{0};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                const int y = next_row.fetch_add(1, std::memory_order_relaxed);
                if (y >= height)
                    return;
                function(y);
            }
        });
    }
    for (std::thread &worker : workers)
        worker.join();
}

static void extract_bright(const Texture2D &source, FloatImage &destination, float threshold, int workers)
{
    parallel_rows(destination.height, workers, [&](int y) {
        for (int x = 0; x < destination.width; ++x) {
            // Kept intentionally as x / width rather than (x + 0.5) / width,
            // matching extractBright in krnls/bloom.cu.
            const float4 color =
                source.sample(static_cast<float>(x) / destination.width, static_cast<float>(y) / destination.height);
            const float luma = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            constexpr float knee = 0.5f;
            float soft = luma - threshold + knee;
            soft = std::fmax(0.0f, std::fmin(soft, 2.0f * knee));
            soft = soft * soft / (4.0f * knee + 0.0001f);
            float weight = std::fmax(soft, luma - threshold) / std::fmax(luma, 0.0001f);
            weight = std::fmin(1.0f, weight);
            destination.write(x, y, make_float4(color.x * weight, color.y * weight, color.z * weight, 1.0f));
        }
    });
}

static void downsample_2x(const Texture2D &source, FloatImage &destination, int workers)
{
    parallel_rows(destination.height, workers, [&](int y) {
        const float dx = 1.0f / (destination.width * 2.0f);
        const float dy = 1.0f / (destination.height * 2.0f);
        for (int x = 0; x < destination.width; ++x) {
            const float u = (x * 2.0f + 0.5f) * dx;
            const float v = (y * 2.0f + 0.5f) * dy;
            const float4 c00 = source.sample(u, v);
            const float4 c10 = source.sample(u + dx, v);
            const float4 c01 = source.sample(u, v + dy);
            const float4 c11 = source.sample(u + dx, v + dy);
            destination.write(x, y, (c00 + c10 + c01 + c11) * 0.25f);
        }
    });
}

static void gaussian_blur_horizontal(const Texture2D &source, FloatImage &destination, float scale, int workers)
{
    constexpr float weights[5] = {0.19638062f, 0.29675293f, 0.09442139f, 0.01037598f, 0.00025940f};
    constexpr float offsets[5] = {0.0f, 1.41176471f, 3.29411765f, 5.17647059f, 7.05882353f};
    parallel_rows(destination.height, workers, [&](int y) {
        const float v = (y + 0.5f) / destination.height;
        for (int x = 0; x < destination.width; ++x) {
            const float u = (x + 0.5f) / destination.width;
            float4 sum = source.sample(u, v) * weights[0];
            for (int index = 1; index < 5; ++index) {
                const float du = offsets[index] * scale / destination.width;
                sum = sum + source.sample(u + du, v) * weights[index];
                sum = sum + source.sample(u - du, v) * weights[index];
            }
            destination.write(x, y, sum);
        }
    });
}

static void gaussian_blur_vertical(const Texture2D &source, FloatImage &destination, float scale, int workers)
{
    constexpr float weights[5] = {0.19638062f, 0.29675293f, 0.09442139f, 0.01037598f, 0.00025940f};
    constexpr float offsets[5] = {0.0f, 1.41176471f, 3.29411765f, 5.17647059f, 7.05882353f};
    parallel_rows(destination.height, workers, [&](int y) {
        const float v = (y + 0.5f) / destination.height;
        for (int x = 0; x < destination.width; ++x) {
            const float u = (x + 0.5f) / destination.width;
            float4 sum = source.sample(u, v) * weights[0];
            for (int index = 1; index < 5; ++index) {
                const float dv = offsets[index] * scale / destination.height;
                sum = sum + source.sample(u, v + dv) * weights[index];
                sum = sum + source.sample(u, v - dv) * weights[index];
            }
            destination.write(x, y, sum);
        }
    });
}

static float4 bicubic_sample(const Texture2D &source, float u, float v, float texture_width, float texture_height)
{
    const float px = u * texture_width - 0.5f;
    const float py = v * texture_height - 0.5f;
    const float ix = std::floor(px);
    const float iy = std::floor(py);
    const float fx = px - ix;
    const float fy = py - iy;
    const float fx2 = fx * fx;
    const float fx3 = fx2 * fx;
    const float fy2 = fy * fy;
    const float fy3 = fy2 * fy;
    const float w0x = (1.0f - fx) * (1.0f - fx) * (1.0f - fx) / 6.0f;
    const float w1x = (3.0f * fx3 - 6.0f * fx2 + 4.0f) / 6.0f;
    const float w2x = (-3.0f * fx3 + 3.0f * fx2 + 3.0f * fx + 1.0f) / 6.0f;
    const float w3x = fx3 / 6.0f;
    const float w0y = (1.0f - fy) * (1.0f - fy) * (1.0f - fy) / 6.0f;
    const float w1y = (3.0f * fy3 - 6.0f * fy2 + 4.0f) / 6.0f;
    const float w2y = (-3.0f * fy3 + 3.0f * fy2 + 3.0f * fy + 1.0f) / 6.0f;
    const float w3y = fy3 / 6.0f;
    const float g0x = w0x + w1x;
    const float g1x = w2x + w3x;
    const float g0y = w0y + w1y;
    const float g1y = w2y + w3y;
    const float h0x = w1x / g0x - 0.5f;
    const float h1x = w3x / g1x + 1.5f;
    const float h0y = w1y / g0y - 0.5f;
    const float h1y = w3y / g1y + 1.5f;
    const float u0 = (ix + h0x + 0.5f) / texture_width;
    const float u1 = (ix + h1x + 0.5f) / texture_width;
    const float v0 = (iy + h0y + 0.5f) / texture_height;
    const float v1 = (iy + h1y + 0.5f) / texture_height;
    const float4 t00 = source.sample(u0, v0);
    const float4 t10 = source.sample(u1, v0);
    const float4 t01 = source.sample(u0, v1);
    const float4 t11 = source.sample(u1, v1);
    const float4 ty0 = t00 * g0x + t10 * g1x;
    const float4 ty1 = t01 * g0x + t11 * g1x;
    return ty0 * g0y + ty1 * g1y;
}

static void composite_bloom(std::uint8_t *output, const Texture2D &original, const std::vector<FloatImage> &levels,
                            int original_width, int original_height, bool use_aces, bool not_use_s_curve, int workers)
{
    constexpr float bloom_weights[8] = {1.0f, 1.5f, 1.0f, 1.5f, 1.8f, 1.0f, 1.0f, 1.0f};
    parallel_rows(original_height, workers, [&](int y) {
        const float v = (y + 0.5f) / original_height;
        for (int x = 0; x < original_width; ++x) {
            const float u = (x + 0.5f) / original_width;
            float4 color = original.sample(u, v);
            float4 bloom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int level = 0; level < static_cast<int>(levels.size()); ++level) {
                const int scale = 1 << (level + 1);
                const float bloom_width = static_cast<float>(original_width / scale);
                const float bloom_height = static_cast<float>(original_height / scale);
                const float4 sample = bicubic_sample(levels[level].texture(), u, v, bloom_width, bloom_height);
                const float weight = level < 8 ? bloom_weights[level] : 1.0f;
                bloom = bloom + sample * weight;
            }
            color = color * 0.55f + bloom * 0.08f;
            color = color * 0.15f;

            if (use_aces) {
                constexpr float a = 2.51f;
                constexpr float b = 0.03f;
                constexpr float c = 2.43f;
                constexpr float d = 0.59f;
                constexpr float e = 0.14f;
                color.x = std::fmin(color.x * (a * color.x + b) / (color.x * (c * color.x + d) + e), 1.0f);
                color.y = std::fmin(color.y * (a * color.y + b) / (color.y * (c * color.y + d) + e), 1.0f);
                color.z = std::fmin(color.z * (a * color.z + b) / (color.z * (c * color.z + d) + e), 1.0f);
            } else {
                color.x = std::pow(color.x, 1.5f);
                color.y = std::pow(color.y, 1.5f);
                color.z = std::pow(color.z, 1.5f);
                color.x = color.x / (1.0f + color.x);
                color.y = color.y / (1.0f + color.y);
                color.z = color.z / (1.0f + color.z);
                color.x = std::pow(color.x, 1.0f / 1.5f);
                color.y = std::pow(color.y, 1.0f / 1.5f);
                color.z = std::pow(color.z, 1.0f / 1.5f);
            }
            if (!not_use_s_curve) {
                color.x = color.x * color.x * (3.0f - 2.0f * color.x);
                color.y = color.y * color.y * (3.0f - 2.0f * color.y);
                color.z = color.z * color.z * (3.0f - 2.0f * color.z);
                color.x = std::pow(color.x, 1.3f);
                color.y = std::pow(color.y, 1.20f);
            }
            const float luma = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            constexpr float saturation = 1.0f;
            color.x = luma + (color.x - luma) * saturation;
            color.y = luma + (color.y - luma) * saturation;
            color.z = luma + (color.z - luma) * saturation;
            color.x = std::fmax(0.0f, std::fmin(color.x * 1.01f, 1.0f));
            color.y = std::fmax(0.0f, std::fmin(color.y * 1.01f, 1.0f));
            color.z = std::fmax(0.0f, std::fmin(color.z * 1.01f, 1.0f));
            constexpr float gamma = 0.7f / 2.2f;
            const std::size_t offset = (static_cast<std::size_t>(y) * original_width + x) * 4;
            output[offset] = static_cast<std::uint8_t>(std::pow(color.x, gamma) * 255.0f);
            output[offset + 1] = static_cast<std::uint8_t>(std::pow(color.y, gamma) * 255.0f);
            output[offset + 2] = static_cast<std::uint8_t>(std::pow(color.z, gamma) * 255.0f);
            output[offset + 3] = 255;
        }
    });
}

static void postprocess_to_rgba8(std::uint8_t *output, const float *raw, const render_params &params,
                                 float bloom_threshold, bool use_aces, bool not_use_s_curve, int workers)
{
    const Texture2D raw_texture{raw, params.imgwidth, params.imgheight, 4, false};
    FloatImage bright(params.imgwidth, params.imgheight, false);
    extract_bright(raw_texture, bright, bloom_threshold, workers);

    const int min_dimension = std::min(params.imgwidth, params.imgheight);
    const int level_count = std::max(1, static_cast<int>(std::nearbyint(std::log2(min_dimension / 8.0f))));
    std::vector<FloatImage> levels;
    std::vector<FloatImage> temporaries;
    levels.reserve(level_count);
    temporaries.reserve(level_count);
    int current_width = params.imgwidth;
    int current_height = params.imgheight;
    for (int level = 0; level < level_count; ++level) {
        current_width = std::max(1, current_width / 2);
        current_height = std::max(1, current_height / 2);
        levels.emplace_back(current_width, current_height);
        temporaries.emplace_back(current_width, current_height);
    }

    Texture2D previous = bright.texture();
    for (int level = 0; level < level_count; ++level) {
        downsample_2x(previous, levels[level], workers);
        gaussian_blur_horizontal(levels[level].texture(), temporaries[level], 1.0f, workers);
        gaussian_blur_vertical(temporaries[level].texture(), levels[level], 1.0f, workers);
        previous = levels[level].texture();
    }
    composite_bloom(output, raw_texture, levels, params.imgwidth, params.imgheight, use_aces, not_use_s_curve, workers);
}

static int render_backend(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                          const render_params &params, int tile_width, int tile_height, int requested_workers,
                          bool use_avx2, bool full_avx_math = false, bool use_avx512 = false)
{
    const int tiles_x = (params.imgwidth + tile_width - 1) / tile_width;
    const int tiles_y = (params.imgheight + tile_height - 1) / tile_height;
    const int total_tiles = tiles_x * tiles_y;
    const int worker_count = effective_worker_count(requested_workers, total_tiles);

    int next_tile = 0;
    std::mutex schedule_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                int tile_index;
                {
                    std::lock_guard<std::mutex> lock(schedule_mutex);
                    if (next_tile >= total_tiles)
                        return;
                    tile_index = next_tile++;
                }
                if (use_avx512) {
                    render_one_tile_avx512(raw, background, disk, color_lut, params, tile_width, tile_height,
                                           tile_index % tiles_x, tile_index / tiles_x);
                } else if (use_avx2 && full_avx_math) {
                    render_one_tile_avx2(raw, background, disk, color_lut, params, tile_width, tile_height,
                                         tile_index % tiles_x, tile_index / tiles_x);
                } else if (use_avx2) {
                    render_one_tile_mixed(raw, background, disk, color_lut, params, tile_width, tile_height,
                                          tile_index % tiles_x, tile_index / tiles_x);
                } else {
                    render_one_tile_scalar(raw, background, disk, color_lut, params, tile_width, tile_height,
                                           tile_index % tiles_x, tile_index / tiles_x);
                }
            }
        });
    }
    for (std::thread &worker : workers)
        worker.join();
    return 0;
}

static int render(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                  const render_params &params, int tile_width, int tile_height, int requested_workers)
{
    if (cpu_supports_avx512())
        return render_backend(raw, background, disk, color_lut, params, tile_width, tile_height, requested_workers,
                              true, true, true);
    if (!cpu_supports_avx2())
        throw std::runtime_error("This CPU or operating system does not support AVX2");
    return render_backend(raw, background, disk, color_lut, params, tile_width, tile_height, requested_workers, true,
                          true);
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static int checked_extent(py::ssize_t value, const char *name)
{
    if (value <= 0 || value > std::numeric_limits<int>::max())
        throw py::value_error(std::string(name) + " must be a non-empty dimension that fits in an int");
    return static_cast<int>(value);
}

static Texture2D texture2d_from_numpy(const FloatArray &array, const char *name)
{
    if (array.ndim() != 3)
        throw py::value_error(std::string(name) + " must have shape (height, width, 3|4)");
    const int height = checked_extent(array.shape(0), name);
    const int width = checked_extent(array.shape(1), name);
    const int channels = checked_extent(array.shape(2), name);
    if (channels != 3 && channels != 4)
        throw py::value_error(std::string(name) + " must have three or four channels");
    return {array.data(), width, height, channels, true};
}

static Texture3D texture3d_from_numpy(const FloatArray &array, const char *name)
{
    if (array.ndim() != 4)
        throw py::value_error(std::string(name) + " must have shape (radius, height, azimuth, 3|4)");
    const int radius = checked_extent(array.shape(0), name);
    const int height = checked_extent(array.shape(1), name);
    const int azimuth = checked_extent(array.shape(2), name);
    const int channels = checked_extent(array.shape(3), name);
    if (channels != 3 && channels != 4)
        throw py::value_error(std::string(name) + " must have three or four channels");
    // The CUDA texture uses (Wrap, Wrap, Clamp) address modes for
    // (phi, z, radius), respectively.
    return {array.data(), radius, height, azimuth, channels, true, true};
}

template <typename T> static T *writable_image_from_numpy(py::array &array, const char *name, int &width, int &height)
{
    if (!array.dtype().is(py::dtype::of<T>()))
        throw py::type_error(std::string(name) + " must have the required NumPy dtype");
    if ((array.flags() & py::array::c_style) != py::array::c_style)
        throw py::type_error(std::string(name) + " must be C-contiguous");
    if (!array.writeable())
        throw py::value_error(std::string(name) + " must be writable");
    if (array.ndim() != 3)
        throw py::value_error(std::string(name) + " must have shape (height, width, 4)");
    height = checked_extent(array.shape(0), name);
    width = checked_extent(array.shape(1), name);
    if (array.shape(2) != 4)
        throw py::value_error(std::string(name) + " must have four channels");
    return static_cast<T *>(array.mutable_data());
}

static render_params params_for_output(const render_params &source, int width, int height, int tile_width,
                                       int tile_height)
{
    if (source.maxstep <= 0 || source.jitternum <= 0 || source.step <= 0.0f || source.focal_length <= 0.0f ||
        tile_width <= 0 || tile_height <= 0)
        throw py::value_error("step, focal_length, maxstep, jitternum, and tile dimensions must be positive");
    const float velocity_squared = source.vfwd * source.vfwd + source.vright * source.vright + source.vup * source.vup;
    if (!std::isfinite(velocity_squared) || velocity_squared >= 1.0f)
        throw py::value_error("velocity magnitude must be smaller than the speed of light");
    render_params params = source;
    params.imgwidth = width;
    params.imgheight = height;
    return params;
}

static void render_into_numpy(py::array raw_img, const FloatArray &background, const FloatArray &prebaked_disk,
                              const FloatArray &color_lut, const render_params &source_params, int tile_width,
                              int tile_height, int workers)
{
    int width;
    int height;
    float *raw = writable_image_from_numpy<float>(raw_img, "raw_img", width, height);
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, width, height, tile_width, tile_height);

    int status;
    {
        py::gil_scoped_release release;
        status =
            render(raw, background_texture, disk_texture, color_lut_texture, params, tile_width, tile_height, workers);
    }
    if (status != 0)
        throw std::runtime_error("CPU renderer failed");
}

static py::array_t<float> render_new_numpy(int width, int height, const FloatArray &background,
                                           const FloatArray &prebaked_disk, const FloatArray &color_lut,
                                           const render_params &params, int tile_width, int tile_height, int workers)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    py::array_t<float> output(py::array::ShapeContainer{static_cast<py::ssize_t>(height),
                                                        static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    render_into_numpy(output, background, prebaked_disk, color_lut, params, tile_width, tile_height, workers);
    return output;
}

// Validation-only reference entry point.  The production Python wrapper uses
// render_new_numpy above; keeping this private binding makes it possible to
// compare the AVX2 packet renderer against the original scalar algorithm.
static py::array_t<float> render_new_scalar_reference_numpy(int width, int height, const FloatArray &background,
                                                            const FloatArray &prebaked_disk,
                                                            const FloatArray &color_lut,
                                                            const render_params &source_params, int tile_width,
                                                            int tile_height, int workers)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    py::array_t<float> output(py::array::ShapeContainer{static_cast<py::ssize_t>(height),
                                                        static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    int output_width;
    int output_height;
    float *raw = writable_image_from_numpy<float>(output, "output", output_width, output_height);
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, output_width, output_height, tile_width, tile_height);
    {
        py::gil_scoped_release release;
        render_backend(raw, background_texture, disk_texture, color_lut_texture, params, tile_width, tile_height,
                       workers, false);
    }
    return output;
}

static py::array_t<float> render_new_accurate_avx_math_numpy(int width, int height, const FloatArray &background,
                                                             const FloatArray &prebaked_disk,
                                                             const FloatArray &color_lut,
                                                             const render_params &source_params, int tile_width,
                                                             int tile_height, int workers)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    if (!cpu_supports_avx2())
        throw std::runtime_error("This CPU or operating system does not support AVX2");
    py::array_t<float> output(py::array::ShapeContainer{static_cast<py::ssize_t>(height),
                                                        static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    int output_width;
    int output_height;
    float *raw = writable_image_from_numpy<float>(output, "output", output_width, output_height);
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, output_width, output_height, tile_width, tile_height);
    {
        py::gil_scoped_release release;
        render_backend(raw, background_texture, disk_texture, color_lut_texture, params, tile_width, tile_height,
                       workers, true, false);
    }
    return output;
}

static py::array_t<float> render_new_full_avx2_math_numpy(int width, int height, const FloatArray &background,
                                                          const FloatArray &prebaked_disk,
                                                          const FloatArray &color_lut,
                                                          const render_params &source_params, int tile_width,
                                                          int tile_height, int workers)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    if (!cpu_supports_avx2())
        throw std::runtime_error("This CPU or operating system does not support AVX2");
    py::array_t<float> output(py::array::ShapeContainer{static_cast<py::ssize_t>(height),
                                                        static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, width, height, tile_width, tile_height);
    {
        py::gil_scoped_release release;
        render_backend(output.mutable_data(), background_texture, disk_texture, color_lut_texture, params, tile_width,
                       tile_height, workers, true, true, false);
    }
    return output;
}

static void render_rgba8_into_numpy(py::array output_img, const FloatArray &background, const FloatArray &prebaked_disk,
                                    const FloatArray &color_lut, const render_params &source_params, int tile_width,
                                    int tile_height, int workers, float bloom_threshold, bool use_aces,
                                    bool not_use_s_curve)
{
    if (!std::isfinite(bloom_threshold))
        throw py::value_error("bloom_threshold must be finite");
    int width;
    int height;
    std::uint8_t *output = writable_image_from_numpy<std::uint8_t>(output_img, "output_img", width, height);
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, width, height, tile_width, tile_height);

    {
        py::gil_scoped_release release;
        FloatImage raw(width, height);
        const int status = render(raw.pixels.data(), background_texture, disk_texture, color_lut_texture, params,
                                  tile_width, tile_height, workers);
        if (status != 0)
            throw std::runtime_error("CPU renderer failed");
        postprocess_to_rgba8(output, raw.pixels.data(), params, bloom_threshold, use_aces, not_use_s_curve, workers);
    }
}

static py::array_t<std::uint8_t> render_rgba8_new_numpy(int width, int height, const FloatArray &background,
                                                        const FloatArray &prebaked_disk, const FloatArray &color_lut,
                                                        const render_params &params, int tile_width, int tile_height,
                                                        int workers, float bloom_threshold, bool use_aces,
                                                        bool not_use_s_curve)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    py::array_t<std::uint8_t> output(py::array::ShapeContainer{
        static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    render_rgba8_into_numpy(output, background, prebaked_disk, color_lut, params, tile_width, tile_height, workers,
                            bloom_threshold, use_aces, not_use_s_curve);
    return output;
}

static py::array_t<std::uint8_t>
render_rgba8_new_accurate_avx_math_numpy(int width, int height, const FloatArray &background,
                                         const FloatArray &prebaked_disk, const FloatArray &color_lut,
                                         const render_params &source_params, int tile_width, int tile_height,
                                         int workers, float bloom_threshold, bool use_aces, bool not_use_s_curve)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    if (!std::isfinite(bloom_threshold))
        throw py::value_error("bloom_threshold must be finite");
    if (!cpu_supports_avx2())
        throw std::runtime_error("This CPU or operating system does not support AVX2");
    py::array_t<std::uint8_t> output(py::array::ShapeContainer{
        static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, width, height, tile_width, tile_height);
    {
        py::gil_scoped_release release;
        FloatImage raw(width, height);
        render_backend(raw.pixels.data(), background_texture, disk_texture, color_lut_texture, params, tile_width,
                       tile_height, workers, true, false);
        postprocess_to_rgba8(output.mutable_data(), raw.pixels.data(), params, bloom_threshold, use_aces,
                             not_use_s_curve, workers);
    }
    return output;
}

static py::array_t<std::uint8_t>
render_rgba8_new_full_avx2_math_numpy(int width, int height, const FloatArray &background,
                                      const FloatArray &prebaked_disk, const FloatArray &color_lut,
                                      const render_params &source_params, int tile_width, int tile_height,
                                      int workers, float bloom_threshold, bool use_aces, bool not_use_s_curve)
{
    if (width <= 0 || height <= 0)
        throw py::value_error("width and height must be positive");
    if (!std::isfinite(bloom_threshold))
        throw py::value_error("bloom_threshold must be finite");
    if (!cpu_supports_avx2())
        throw std::runtime_error("This CPU or operating system does not support AVX2");
    py::array_t<std::uint8_t> output(py::array::ShapeContainer{
        static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    const Texture2D background_texture = texture2d_from_numpy(background, "background");
    const Texture3D disk_texture = texture3d_from_numpy(prebaked_disk, "prebaked_disk");
    const Texture2D color_lut_texture = texture2d_from_numpy(color_lut, "color_lut");
    const render_params params = params_for_output(source_params, width, height, tile_width, tile_height);
    {
        py::gil_scoped_release release;
        FloatImage raw(width, height);
        render_backend(raw.pixels.data(), background_texture, disk_texture, color_lut_texture, params, tile_width,
                       tile_height, workers, true, true, false);
        postprocess_to_rgba8(output.mutable_data(), raw.pixels.data(), params, bloom_threshold, use_aces,
                             not_use_s_curve, workers);
    }
    return output;
}

PYBIND11_MODULE(cpu_render_native, module)
{
    module.doc() = "Full-vector CPU black-hole renderer with automatic AVX-512/AVX2 dispatch.";

    py::class_<render_params>(module, "RenderParams")
        .def(py::init<>())
        .def_readwrite("time", &render_params::time)
        .def_readwrite("cam_pos_x", &render_params::cam_pos_x)
        .def_readwrite("cam_pos_y", &render_params::cam_pos_y)
        .def_readwrite("cam_pos_z", &render_params::cam_pos_z)
        .def_readwrite("fwd_x", &render_params::fwd_x)
        .def_readwrite("fwd_y", &render_params::fwd_y)
        .def_readwrite("fwd_z", &render_params::fwd_z)
        .def_readwrite("right_x", &render_params::right_x)
        .def_readwrite("right_y", &render_params::right_y)
        .def_readwrite("right_z", &render_params::right_z)
        .def_readwrite("up_x", &render_params::up_x)
        .def_readwrite("up_y", &render_params::up_y)
        .def_readwrite("up_z", &render_params::up_z)
        .def_readwrite("vfwd", &render_params::vfwd)
        .def_readwrite("vright", &render_params::vright)
        .def_readwrite("vup", &render_params::vup)
        .def_readwrite("physwidth", &render_params::physwidth)
        .def_readwrite("physheight", &render_params::physheight)
        .def_readwrite("focal_length", &render_params::focal_length)
        .def_readwrite("step", &render_params::step)
        .def_readwrite("maxstep", &render_params::maxstep)
        .def_readwrite("jitternum", &render_params::jitternum)
        .def_readwrite("frames", &render_params::frames)
        .def_readwrite("flags", &render_params::flags);

    module.def("render_into", &render_into_numpy, py::arg("raw_img"), py::arg("background"), py::arg("prebaked_disk"),
               py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32, py::arg("tile_height") = 8,
               py::arg("workers") = 0, "Render directly into a C-contiguous float32 (height, width, 4) NumPy array.");
    module.def("render_new", &render_new_numpy, py::arg("width"), py::arg("height"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32,
               py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Allocate and return an automatically dispatched AVX-512/AVX2 float32 image.");
    module.def("_render_new_scalar_reference", &render_new_scalar_reference_numpy, py::arg("width"), py::arg("height"),
               py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Validation-only scalar reference renderer.");
    module.def("_render_new_avx2_accurate", &render_new_accurate_avx_math_numpy, py::arg("width"), py::arg("height"),
               py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Validation-only AVX2 renderer with accurate scalar texture and transcendental functions.");
    module.def("_render_new_avx2_full_math", &render_new_full_avx2_math_numpy, py::arg("width"), py::arg("height"),
               py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Validation-only forced full-AVX2 render path.");
    module.def("avx2_supported", &cpu_supports_avx2, "Return whether AVX2 execution is available.");
    module.def("avx512_supported", &cpu_supports_avx512,
               "Return whether AVX-512F/DQ execution and OS ZMM state are available.");
    module.def("render_rgba8_into", &render_rgba8_into_numpy, py::arg("output_img"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32,
               py::arg("tile_height") = 8, py::arg("workers") = 0, py::arg("bloom_threshold") = 12.0f,
               py::arg("use_aces") = true, py::arg("not_use_s_curve") = false,
               "Render and post-process into a C-contiguous uint8 (height, width, 4) NumPy array.");
    module.def("render_rgba8_new", &render_rgba8_new_numpy, py::arg("width"), py::arg("height"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32,
               py::arg("tile_height") = 8, py::arg("workers") = 0, py::arg("bloom_threshold") = 12.0f,
               py::arg("use_aces") = true, py::arg("not_use_s_curve") = false,
               "Allocate, render with automatic AVX-512/AVX2 dispatch, and return a post-processed uint8 image.");
    module.def("_render_rgba8_new_avx2_accurate", &render_rgba8_new_accurate_avx_math_numpy, py::arg("width"),
               py::arg("height"), py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"),
               py::arg("params"), py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               py::arg("bloom_threshold") = 12.0f, py::arg("use_aces") = true, py::arg("not_use_s_curve") = false,
               "Validation-only accurate AVX2 render and post-process entry point.");
    module.def("_render_rgba8_new_avx2_full_math", &render_rgba8_new_full_avx2_math_numpy, py::arg("width"),
               py::arg("height"),
               py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               py::arg("bloom_threshold") = 12.0f, py::arg("use_aces") = true, py::arg("not_use_s_curve") = false,
               "Validation-only forced full-AVX2 RGBA8 path.");
}
