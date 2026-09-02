// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Qianyv-Li8262
// Sixteen-ray AVX-512 implementation. Texture gathers and transcendental
// approximations remain fully vectorized, matching the public AVX2 algorithm.

#include "cpu_render_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Sixteen independent rays are stored structure-of-arrays style.  These helpers
// deliberately use AVX-512 intrinsics: the packet path does not depend on LLVM's
// loop or SLP auto-vectorizers.
struct float2x16 {
    __m512 x;
    __m512 y;
};

struct float3x16 {
    __m512 x;
    __m512 y;
    __m512 z;
};

struct float4x16 {
    __m512 x;
    __m512 y;
    __m512 z;
    __m512 w;
};

static inline __m512 v16(float value)
{
    return _mm512_set1_ps(value);
}

// AVX-512 comparisons produce compact k-masks. The rest of this file keeps
// masks as full vectors so the AVX2 algorithm can remain structurally identical.
#define v16_cmp_ps(a, b, predicate) \
    _mm512_castsi512_ps(_mm512_movm_epi32(_mm512_cmp_ps_mask((a), (b), (predicate))))

static inline __m512i i16_cmpgt_epi32(__m512i a, __m512i b)
{
    return _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(a, b));
}

static inline int v16_movemask_ps(__m512 mask)
{
    return static_cast<int>(_mm512_movepi32_mask(_mm512_castps_si512(mask)));
}

#define v16_mask_i32gather_ps(source, base, indices, mask, scale) \
    _mm512_mask_i32gather_ps((source), static_cast<__mmask16>(v16_movemask_ps(mask)), (indices), (base), (scale))

static inline __m512 v16_abs(__m512 value)
{
    return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value);
}

static inline __m512 v16_select(__m512 mask, __m512 when_true, __m512 when_false)
{
    return _mm512_or_ps(_mm512_and_ps(mask, when_true), _mm512_andnot_ps(mask, when_false));
}

static inline __m512 v16_clamp(__m512 value, float low, float high)
{
    return _mm512_min_ps(_mm512_max_ps(value, v16(low)), v16(high));
}

static inline float3x16 v3_add(float3x16 a, float3x16 b)
{
    return {_mm512_add_ps(a.x, b.x), _mm512_add_ps(a.y, b.y), _mm512_add_ps(a.z, b.z)};
}

static inline float3x16 v3_sub(float3x16 a, float3x16 b)
{
    return {_mm512_sub_ps(a.x, b.x), _mm512_sub_ps(a.y, b.y), _mm512_sub_ps(a.z, b.z)};
}

static inline float3x16 v3_mul(float3x16 value, __m512 scalar)
{
    return {_mm512_mul_ps(value.x, scalar), _mm512_mul_ps(value.y, scalar), _mm512_mul_ps(value.z, scalar)};
}

static inline __m512 v3_dot(float3x16 a, float3x16 b)
{
    return _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(a.x, b.x), _mm512_mul_ps(a.y, b.y)), _mm512_mul_ps(a.z, b.z));
}

static inline __m512 v3_length(float3x16 value)
{
    return _mm512_sqrt_ps(v3_dot(value, value));
}

static inline float3x16 v3_normalize(float3x16 value)
{
    return v3_mul(value, _mm512_div_ps(v16(1.0f), v3_length(value)));
}

static inline float4x16 v4_zero()
{
    const __m512 zero = _mm512_setzero_ps();
    return {zero, zero, zero, zero};
}

static inline float4x16 v4_add(float4x16 a, float4x16 b)
{
    return {_mm512_add_ps(a.x, b.x), _mm512_add_ps(a.y, b.y), _mm512_add_ps(a.z, b.z), _mm512_add_ps(a.w, b.w)};
}

static inline float4x16 v4_mul(float4x16 value, __m512 scalar)
{
    return {_mm512_mul_ps(value.x, scalar), _mm512_mul_ps(value.y, scalar), _mm512_mul_ps(value.z, scalar),
            _mm512_mul_ps(value.w, scalar)};
}

static inline __m512 valid_lane_mask(int lane_count)
{
    return _mm512_castsi512_ps(_mm512_setr_epi32(
        lane_count > 0 ? -1 : 0, lane_count > 1 ? -1 : 0, lane_count > 2 ? -1 : 0, lane_count > 3 ? -1 : 0,
        lane_count > 4 ? -1 : 0, lane_count > 5 ? -1 : 0, lane_count > 6 ? -1 : 0, lane_count > 7 ? -1 : 0,
        lane_count > 8 ? -1 : 0, lane_count > 9 ? -1 : 0, lane_count > 10 ? -1 : 0, lane_count > 11 ? -1 : 0,
        lane_count > 12 ? -1 : 0, lane_count > 13 ? -1 : 0, lane_count > 14 ? -1 : 0, lane_count > 15 ? -1 : 0));
}

static inline bool any_lane(__m512 mask)
{
    return v16_movemask_ps(mask) != 0;
}

static inline void store4(float4x16 value, float *x, float *y, float *z, float *w)
{
    _mm512_store_ps(x, value.x);
    _mm512_store_ps(y, value.y);
    _mm512_store_ps(z, value.z);
    _mm512_store_ps(w, value.w);
}

static inline float4x16 v4_sub(float4x16 a, float4x16 b)
{
    return {_mm512_sub_ps(a.x, b.x), _mm512_sub_ps(a.y, b.y), _mm512_sub_ps(a.z, b.z), _mm512_sub_ps(a.w, b.w)};
}

static inline float4x16 v4_select(__m512 mask, float4x16 when_true, float4x16 when_false)
{
    return {v16_select(mask, when_true.x, when_false.x), v16_select(mask, when_true.y, when_false.y),
            v16_select(mask, when_true.z, when_false.z), v16_select(mask, when_true.w, when_false.w)};
}

static inline __m512i i16_select(__m512i mask, __m512i when_true, __m512i when_false)
{
    return _mm512_or_si512(_mm512_and_si512(mask, when_true), _mm512_andnot_si512(mask, when_false));
}

static inline __m512i i16_clamp(__m512i value, int low, int high)
{
    return _mm512_min_epi32(_mm512_max_epi32(value, _mm512_set1_epi32(low)), _mm512_set1_epi32(high));
}

static inline __m512i i16_wrap_once(__m512i value, int extent)
{
    const __m512i zero = _mm512_setzero_si512();
    const __m512i size = _mm512_set1_epi32(extent);
    value = i16_select(i16_cmpgt_epi32(zero, value), _mm512_add_epi32(value, size), value);
    return i16_select(i16_cmpgt_epi32(value, _mm512_set1_epi32(extent - 1)), _mm512_sub_epi32(value, size), value);
}

static inline __m512i pcg_hash_v16(__m512i input)
{
    const __m512i state = _mm512_add_epi32(_mm512_mullo_epi32(input, _mm512_set1_epi32(747796405)),
                                           _mm512_set1_epi32(static_cast<int>(2891336453u)));
    const __m512i shift = _mm512_add_epi32(_mm512_srli_epi32(state, 28), _mm512_set1_epi32(4));
    const __m512i word =
        _mm512_mullo_epi32(_mm512_xor_si512(_mm512_srlv_epi32(state, shift), state), _mm512_set1_epi32(277803737));
    return _mm512_xor_si512(_mm512_srli_epi32(word, 22), word);
}

static inline __m512i reverse_bits_v16(__m512i value)
{
    value = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(value, _mm512_set1_epi32(0x55555555)), 1),
                            _mm512_and_si512(_mm512_srli_epi32(value, 1), _mm512_set1_epi32(0x55555555)));
    value = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(value, _mm512_set1_epi32(0x33333333)), 2),
                            _mm512_and_si512(_mm512_srli_epi32(value, 2), _mm512_set1_epi32(0x33333333)));
    value = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(value, _mm512_set1_epi32(0x0f0f0f0f)), 4),
                            _mm512_and_si512(_mm512_srli_epi32(value, 4), _mm512_set1_epi32(0x0f0f0f0f)));
    value = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(value, _mm512_set1_epi32(0x00ff00ff)), 8),
                            _mm512_and_si512(_mm512_srli_epi32(value, 8), _mm512_set1_epi32(0x00ff00ff)));
    return _mm512_or_si512(_mm512_slli_epi32(value, 16), _mm512_srli_epi32(value, 16));
}

static inline __m512 uint32_to_unit_float_v16(__m512i value)
{
    const __m512 high31 = _mm512_cvtepi32_ps(_mm512_srli_epi32(value, 1));
    const __m512 low_bit = _mm512_cvtepi32_ps(_mm512_and_si512(value, _mm512_set1_epi32(1)));
    return _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(high31, high31), low_bit), v16(2.3283064365386963e-10f));
}

static inline __m512 rand_float_v16(__m512i seed)
{
    seed = _mm512_xor_si512(_mm512_xor_si512(seed, _mm512_set1_epi32(61)), _mm512_srli_epi32(seed, 16));
    seed = _mm512_mullo_epi32(seed, _mm512_set1_epi32(9));
    seed = _mm512_xor_si512(seed, _mm512_srli_epi32(seed, 4));
    seed = _mm512_mullo_epi32(seed, _mm512_set1_epi32(static_cast<int>(0x27d4eb2du)));
    seed = _mm512_xor_si512(seed, _mm512_srli_epi32(seed, 15));
    return uint32_to_unit_float_v16(seed);
}

static inline float2x16 hammersley_v16(int index, int count, __m512i pixel_x, __m512i pixel_y, std::uint32_t frames)
{
    const __m512 h_x = v16(static_cast<float>(index) / static_cast<float>(count));
    const __m512 h_y = _mm512_mul_ps(uint32_to_unit_float_v16(reverse_bits_v16(_mm512_set1_epi32(index))), v16(1.0f));
    const __m512i frame_seed = _mm512_set1_epi32(static_cast<int>(frames * 1919810u));
    const __m512 shift_x = rand_float_v16(_mm512_xor_si512(_mm512_xor_si512(pixel_x, pcg_hash_v16(pixel_y)), frame_seed));
    const __m512 shift_y = rand_float_v16(_mm512_xor_si512(
        _mm512_xor_si512(pixel_y, pcg_hash_v16(_mm512_mullo_epi32(pixel_x, _mm512_set1_epi32(114514)))), frame_seed));
    const __m512 sum_x = _mm512_add_ps(h_x, shift_x);
    const __m512 sum_y = _mm512_add_ps(h_y, shift_y);
    return {_mm512_sub_ps(sum_x, _mm512_floor_ps(sum_x)), _mm512_sub_ps(sum_y, _mm512_floor_ps(sum_y))};
}

// Cephes-derived exp approximation with range reduction to powers of two.
// Its relative error is suitable for the opacity calculation and it stays
// entirely in eight AVX2 lanes.
static inline __m512 exp_v16(__m512 value)
{
    value = v16_clamp(value, -88.3762626647949f, 88.3762626647949f);
    __m512 exponent = _mm512_add_ps(_mm512_mul_ps(value, v16(1.44269504088896341f)), v16(0.5f));
    exponent = _mm512_floor_ps(exponent);
    value = _mm512_sub_ps(value, _mm512_mul_ps(exponent, v16(0.693359375f)));
    value = _mm512_sub_ps(value, _mm512_mul_ps(exponent, v16(-2.12194440e-4f)));
    const __m512 squared = _mm512_mul_ps(value, value);

    __m512 polynomial = v16(1.9875691500e-4f);
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, value), v16(1.3981999507e-3f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, value), v16(8.3334519073e-3f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, value), v16(4.1665795894e-2f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, value), v16(1.6666665459e-1f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, value), v16(5.0000001201e-1f));
    polynomial = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(polynomial, squared), value), v16(1.0f));

    __m512i power = _mm512_cvttps_epi32(exponent);
    power = _mm512_slli_epi32(_mm512_add_epi32(power, _mm512_set1_epi32(127)), 23);
    return _mm512_mul_ps(polynomial, _mm512_castsi512_ps(power));
}

static inline __m512 atan_reduced_v16(__m512 value)
{
    const __m512 squared = _mm512_mul_ps(value, value);
    __m512 polynomial = v16(8.05374449538e-2f);
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, squared), v16(-1.38776856032e-1f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, squared), v16(1.99777106478e-1f));
    polynomial = _mm512_add_ps(_mm512_mul_ps(polynomial, squared), v16(-3.33329491539e-1f));
    return _mm512_add_ps(value, _mm512_mul_ps(_mm512_mul_ps(value, squared), polynomial));
}

static inline __m512 atan_unit_v16(__m512 value)
{
    const __m512 reduce = v16_cmp_ps(value, v16(0.4142135623730950f), _CMP_GT_OQ);
    const __m512 transformed = _mm512_div_ps(_mm512_sub_ps(value, v16(1.0f)), _mm512_add_ps(value, v16(1.0f)));
    return v16_select(reduce, _mm512_add_ps(v16(0.7853981633974483f), atan_reduced_v16(transformed)),
                     atan_reduced_v16(value));
}

static inline __m512 atan2_v16(__m512 y, __m512 x)
{
    const __m512 abs_x = v16_abs(x);
    const __m512 abs_y = v16_abs(y);
    const __m512 y_larger = v16_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
    const __m512 maximum = _mm512_max_ps(abs_x, abs_y);
    const __m512 minimum = _mm512_min_ps(abs_x, abs_y);
    const __m512 nonzero = v16_cmp_ps(maximum, _mm512_setzero_ps(), _CMP_GT_OQ);
    const __m512 ratio = v16_select(nonzero, _mm512_div_ps(minimum, maximum), _mm512_setzero_ps());
    __m512 angle = atan_unit_v16(ratio);
    angle = v16_select(y_larger, _mm512_sub_ps(v16(1.5707963267948966f), angle), angle);
    angle = v16_select(v16_cmp_ps(x, _mm512_setzero_ps(), _CMP_LT_OQ), _mm512_sub_ps(v16(3.1415926535897932f), angle),
                      angle);
    return _mm512_xor_ps(angle, _mm512_and_ps(y, _mm512_set1_ps(-0.0f)));
}

static inline __m512 asin_v16(__m512 value)
{
    value = v16_clamp(value, -1.0f, 1.0f);
    const __m512 adjacent = _mm512_sqrt_ps(_mm512_max_ps(
        _mm512_setzero_ps(), _mm512_mul_ps(_mm512_sub_ps(v16(1.0f), value), _mm512_add_ps(v16(1.0f), value))));
    return atan2_v16(value, adjacent);
}

static inline __m512 fast_mod2pi_v16(__m512 value)
{
    return _mm512_sub_ps(value,
                         _mm512_mul_ps(_mm512_floor_ps(_mm512_mul_ps(value, v16(0.159154943f))), v16(6.283185307f)));
}

static inline float4x16 gather_texels(const float *data, __m512i texel_index, int channels, __m512 mask)
{
    const __m512i base = _mm512_mullo_epi32(texel_index, _mm512_set1_epi32(channels));
    const __m512 zero = _mm512_setzero_ps();
    const __m512 x = v16_mask_i32gather_ps(zero, data, base, mask, 4);
    const __m512 y = v16_mask_i32gather_ps(zero, data, _mm512_add_epi32(base, _mm512_set1_epi32(1)), mask, 4);
    const __m512 z = v16_mask_i32gather_ps(zero, data, _mm512_add_epi32(base, _mm512_set1_epi32(2)), mask, 4);
    const __m512 w = channels == 4
                         ? v16_mask_i32gather_ps(zero, data, _mm512_add_epi32(base, _mm512_set1_epi32(3)), mask, 4)
                         : zero;
    return {x, y, z, w};
}

static inline float4x16 lerp_v4(float4x16 a, float4x16 b, __m512 fraction)
{
    return v4_add(a, v4_mul(v4_sub(b, a), fraction));
}

static inline float4x16 sample_texture2d_v16(const Texture2D &texture, __m512 u, __m512 v, __m512 mask)
{
    const __m512 x = _mm512_sub_ps(_mm512_mul_ps(u, v16(static_cast<float>(texture.width))), v16(0.5f));
    const __m512 y = _mm512_sub_ps(_mm512_mul_ps(v, v16(static_cast<float>(texture.height))), v16(0.5f));
    const __m512 floor_x = _mm512_floor_ps(x);
    const __m512 floor_y = _mm512_floor_ps(y);
    const __m512 tx = _mm512_sub_ps(x, floor_x);
    const __m512 ty = _mm512_sub_ps(y, floor_y);
    const __m512i x0 = i16_clamp(_mm512_cvttps_epi32(floor_x), 0, texture.width - 1);
    const __m512i x1 =
        i16_clamp(_mm512_add_epi32(_mm512_cvttps_epi32(floor_x), _mm512_set1_epi32(1)), 0, texture.width - 1);
    const __m512i y0 = i16_clamp(_mm512_cvttps_epi32(floor_y), 0, texture.height - 1);
    const __m512i y1 =
        i16_clamp(_mm512_add_epi32(_mm512_cvttps_epi32(floor_y), _mm512_set1_epi32(1)), 0, texture.height - 1);
    const __m512i row0 = _mm512_mullo_epi32(y0, _mm512_set1_epi32(texture.width));
    const __m512i row1 = _mm512_mullo_epi32(y1, _mm512_set1_epi32(texture.width));
    const float4x16 top = lerp_v4(gather_texels(texture.data, _mm512_add_epi32(row0, x0), texture.channels, mask),
                                 gather_texels(texture.data, _mm512_add_epi32(row0, x1), texture.channels, mask), tx);
    const float4x16 bottom =
        lerp_v4(gather_texels(texture.data, _mm512_add_epi32(row1, x0), texture.channels, mask),
                gather_texels(texture.data, _mm512_add_epi32(row1, x1), texture.channels, mask), tx);
    return lerp_v4(top, bottom, ty);
}

static inline float4x16 sample_texture3d_v16(const Texture3D &texture, __m512 u, __m512 v, __m512 w, __m512 mask)
{
    const __m512 x = _mm512_sub_ps(_mm512_mul_ps(u, v16(static_cast<float>(texture.azimuth))), v16(0.5f));
    const __m512 y = _mm512_sub_ps(_mm512_mul_ps(v, v16(static_cast<float>(texture.height))), v16(0.5f));
    const __m512 z = _mm512_sub_ps(_mm512_mul_ps(w, v16(static_cast<float>(texture.radius))), v16(0.5f));
    const __m512 floor_x = _mm512_floor_ps(x);
    const __m512 floor_y = _mm512_floor_ps(y);
    const __m512 floor_z = _mm512_floor_ps(z);
    const __m512 tx = _mm512_sub_ps(x, floor_x);
    const __m512 ty = _mm512_sub_ps(y, floor_y);
    const __m512 tz = _mm512_sub_ps(z, floor_z);
    const __m512i raw_x0 = _mm512_cvttps_epi32(floor_x);
    const __m512i raw_y0 = _mm512_cvttps_epi32(floor_y);
    const __m512i raw_z0 = _mm512_cvttps_epi32(floor_z);
    const __m512i x0 = i16_wrap_once(raw_x0, texture.azimuth);
    const __m512i x1 = i16_wrap_once(_mm512_add_epi32(raw_x0, _mm512_set1_epi32(1)), texture.azimuth);
    const __m512i y0 = i16_wrap_once(raw_y0, texture.height);
    const __m512i y1 = i16_wrap_once(_mm512_add_epi32(raw_y0, _mm512_set1_epi32(1)), texture.height);
    const __m512i z0 = i16_clamp(raw_z0, 0, texture.radius - 1);
    const __m512i z1 = i16_clamp(_mm512_add_epi32(raw_z0, _mm512_set1_epi32(1)), 0, texture.radius - 1);
    const __m512i slice = _mm512_set1_epi32(texture.height * texture.azimuth);
    const __m512i row_width = _mm512_set1_epi32(texture.azimuth);
    const __m512i z0_base = _mm512_mullo_epi32(z0, slice);
    const __m512i z1_base = _mm512_mullo_epi32(z1, slice);
    const __m512i y0_base = _mm512_mullo_epi32(y0, row_width);
    const __m512i y1_base = _mm512_mullo_epi32(y1, row_width);

    const float4x16 c000 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z0_base, y0_base), x0), texture.channels, mask);
    const float4x16 c100 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z0_base, y0_base), x1), texture.channels, mask);
    const float4x16 c010 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z0_base, y1_base), x0), texture.channels, mask);
    const float4x16 c110 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z0_base, y1_base), x1), texture.channels, mask);
    const float4x16 c001 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z1_base, y0_base), x0), texture.channels, mask);
    const float4x16 c101 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z1_base, y0_base), x1), texture.channels, mask);
    const float4x16 c011 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z1_base, y1_base), x0), texture.channels, mask);
    const float4x16 c111 =
        gather_texels(texture.data, _mm512_add_epi32(_mm512_add_epi32(z1_base, y1_base), x1), texture.channels, mask);
    const float4x16 c00 = lerp_v4(c000, c100, tx);
    const float4x16 c10 = lerp_v4(c010, c110, tx);
    const float4x16 c01 = lerp_v4(c001, c101, tx);
    const float4x16 c11 = lerp_v4(c011, c111, tx);
    return lerp_v4(lerp_v4(c00, c10, ty), lerp_v4(c01, c11, ty), tz);
}

static inline __m512 smoothstep_v16(float edge0, float edge1, __m512 value)
{
    const __m512 t = v16_clamp(_mm512_div_ps(_mm512_sub_ps(value, v16(edge0)), v16(edge1 - edge0)), 0.0f, 1.0f);
    return _mm512_mul_ps(_mm512_mul_ps(t, t), _mm512_sub_ps(v16(3.0f), _mm512_mul_ps(v16(2.0f), t)));
}

static inline float3x16 wavelength_to_rgb_v16(__m512 lambda_nm)
{
    const __m512 uv_to_visible = smoothstep_v16(330.0f, 410.0f, lambda_nm);
    const __m512 visible_to_ir = smoothstep_v16(720.0f, 860.0f, lambda_nm);
    const __m512 uv_weight = _mm512_sub_ps(v16(1.0f), uv_to_visible);
    const __m512 ir_weight = visible_to_ir;
    const __m512 visible_weight = _mm512_mul_ps(uv_to_visible, _mm512_sub_ps(v16(1.0f), visible_to_ir));
    const __m512 wavelength = v16_clamp(lambda_nm, 380.0f, 780.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = v16(1.0f);
    __m512 red = zero;
    __m512 green = zero;
    __m512 blue = zero;

    const __m512 range0 = v16_cmp_ps(wavelength, v16(440.0f), _CMP_LT_OQ);
    const __m512 range1 = _mm512_andnot_ps(range0, v16_cmp_ps(wavelength, v16(490.0f), _CMP_LT_OQ));
    const __m512 below510 = v16_cmp_ps(wavelength, v16(510.0f), _CMP_LT_OQ);
    const __m512 range2 = _mm512_andnot_ps(_mm512_or_ps(range0, range1), below510);
    const __m512 below580 = v16_cmp_ps(wavelength, v16(580.0f), _CMP_LT_OQ);
    const __m512 range3 = _mm512_andnot_ps(_mm512_or_ps(_mm512_or_ps(range0, range1), range2), below580);
    const __m512 below645 = v16_cmp_ps(wavelength, v16(645.0f), _CMP_LT_OQ);
    const __m512 prior3 = _mm512_or_ps(_mm512_or_ps(range0, range1), _mm512_or_ps(range2, range3));
    const __m512 range4 = _mm512_andnot_ps(prior3, below645);
    const __m512 range5 = _mm512_andnot_ps(_mm512_or_ps(prior3, range4), _mm512_castsi512_ps(_mm512_set1_epi32(-1)));

    red = v16_select(range0, _mm512_div_ps(_mm512_sub_ps(v16(440.0f), wavelength), v16(60.0f)), red);
    blue = v16_select(range0, one, blue);
    green = v16_select(range1, _mm512_div_ps(_mm512_sub_ps(wavelength, v16(440.0f)), v16(50.0f)), green);
    blue = v16_select(range1, one, blue);
    green = v16_select(range2, one, green);
    blue = v16_select(range2, _mm512_div_ps(_mm512_sub_ps(v16(510.0f), wavelength), v16(20.0f)), blue);
    red = v16_select(range3, _mm512_div_ps(_mm512_sub_ps(wavelength, v16(510.0f)), v16(70.0f)), red);
    green = v16_select(range3, one, green);
    red = v16_select(range4, one, red);
    green = v16_select(range4, _mm512_div_ps(_mm512_sub_ps(v16(645.0f), wavelength), v16(65.0f)), green);
    red = v16_select(range5, one, red);

    __m512 edge_factor = one;
    const __m512 violet_edge = _mm512_add_ps(
        v16(0.3f), _mm512_mul_ps(v16(0.7f), _mm512_div_ps(_mm512_sub_ps(wavelength, v16(380.0f)), v16(40.0f))));
    const __m512 red_edge = _mm512_add_ps(
        v16(0.3f), _mm512_mul_ps(v16(0.7f), _mm512_div_ps(_mm512_sub_ps(v16(780.0f), wavelength), v16(80.0f))));
    edge_factor = v16_select(v16_cmp_ps(wavelength, v16(420.0f), _CMP_LT_OQ), violet_edge, edge_factor);
    edge_factor = v16_select(v16_cmp_ps(wavelength, v16(700.0f), _CMP_GT_OQ), red_edge, edge_factor);

    const float3x16 visible_rgb{_mm512_mul_ps(red, edge_factor), _mm512_mul_ps(green, edge_factor),
                               _mm512_mul_ps(blue, edge_factor)};
    const float3x16 outside_uv{v16(0.2475f), zero, v16(0.45f)};
    const float3x16 outside_ir{v16(0.45f), zero, zero};
    return v3_add(v3_add(v3_mul(visible_rgb, visible_weight), v3_mul(outside_uv, uv_weight)),
                  v3_mul(outside_ir, ir_weight));
}

static inline float3x16 rgb_three_line_frequency_shift_v16(float3x16 rgb, __m512 g)
{
    g = _mm512_max_ps(g, v16(1e-6f));
    const float3x16 c_red = wavelength_to_rgb_v16(_mm512_div_ps(v16(700.0f), g));
    const float3x16 c_green = wavelength_to_rgb_v16(_mm512_div_ps(v16(546.1f), g));
    const float3x16 c_blue = wavelength_to_rgb_v16(_mm512_div_ps(v16(435.8f), g));
    constexpr float green_leaks_to_red = (546.1f - 510.0f) / 70.0f;
    constexpr float blue_leaks_to_red = (440.0f - 435.8f) / 60.0f;
    const float3x16 coefficients{_mm512_sub_ps(_mm512_sub_ps(rgb.x, _mm512_mul_ps(v16(green_leaks_to_red), rgb.y)),
                                              _mm512_mul_ps(v16(blue_leaks_to_red), rgb.z)),
                                rgb.y, rgb.z};
    float3x16 output =
        v3_add(v3_add(v3_mul(c_red, coefficients.x), v3_mul(c_green, coefficients.y)), v3_mul(c_blue, coefficients.z));
    const __m512 g3 = _mm512_mul_ps(_mm512_mul_ps(g, g), g);
    output = v3_mul(output, g3);
    output.x = _mm512_max_ps(output.x, _mm512_setzero_ps());
    output.y = _mm512_max_ps(output.y, _mm512_setzero_ps());
    output.z = _mm512_max_ps(output.z, _mm512_setzero_ps());
    return output;
}

static inline float4x16 disk_emission_dep_v16(__m512 temperature, __m512 intensity, const Texture2D &color_lut,
                                            __m512 mask)
{
    const float4x16 color = sample_texture2d_v16(
        color_lut, _mm512_div_ps(_mm512_sub_ps(temperature, v16(510.0f)), v16(20000.0f)), v16(0.5f), mask);
    return {_mm512_mul_ps(color.x, intensity), _mm512_mul_ps(color.y, intensity), _mm512_mul_ps(color.z, intensity),
            v16(1.0f)};
}

void render_one_tile_avx512(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                          const render_params &params, int tile_width, int tile_height, int tile_x, int tile_y)
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

    const float3x16 e0v{v16(e0.x), v16(e0.y), v16(e0.z)};
    const float3x16 e1v{v16(e1_fwd.x), v16(e1_fwd.y), v16(e1_fwd.z)};
    const float3x16 e2v{v16(e2_right.x), v16(e2_right.y), v16(e2_right.z)};
    const float3x16 e3v{v16(e3_up.x), v16(e3_up.y), v16(e3_up.z)};
    const float3x16 initial_pos_v{v16(initial_cam_pos.x), v16(initial_cam_pos.y), v16(initial_cam_pos.z)};
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = v16(1.0f);

    alignas(64) float lane_a[16];
    alignas(64) float lane_b[16];
    alignas(64) float lane_c[16];
    alignas(64) float lane_d[16];

    for (int pixel_y = y_begin; pixel_y < y_end; ++pixel_y) {
        for (int packet_x = x_begin; packet_x < x_end; packet_x += 16) {
            const int lane_count = std::min(16, x_end - packet_x);
            const __m512 valid_mask = valid_lane_mask(lane_count);
            const __m512i pixel_x_v = _mm512_setr_epi32(
                packet_x, packet_x + 1, packet_x + 2, packet_x + 3, packet_x + 4, packet_x + 5, packet_x + 6,
                packet_x + 7, packet_x + 8, packet_x + 9, packet_x + 10, packet_x + 11, packet_x + 12,
                packet_x + 13, packet_x + 14, packet_x + 15);
            const __m512i pixel_y_v = _mm512_set1_epi32(pixel_y);
            float4x16 buffer = v4_zero();

            for (int sample = 0; sample < params.jitternum; ++sample) {
                const float2x16 jitter = hammersley_v16(sample, params.jitternum, pixel_x_v, pixel_y_v, 1u);
                const __m512 physical_x =
                    _mm512_mul_ps(_mm512_sub_ps(_mm512_div_ps(_mm512_add_ps(_mm512_cvtepi32_ps(pixel_x_v), jitter.x),
                                                              v16(static_cast<float>(params.imgwidth))),
                                                v16(0.5f)),
                                  v16(params.physwidth));
                const __m512 physical_y =
                    _mm512_mul_ps(_mm512_sub_ps(_mm512_div_ps(_mm512_add_ps(_mm512_cvtepi32_ps(pixel_y_v), jitter.y),
                                                              v16(static_cast<float>(params.imgheight))),
                                                v16(0.5f)),
                                  v16(params.physheight));

                const float3x16 camera_ray =
                    v3_normalize({v16(params.focal_length), physical_x, _mm512_sub_ps(zero, physical_y)});
                float3x16 cam_pos = initial_pos_v;
                __m512 delta_t = zero;
                __m512 radius = v16(camera_radius);
                __m512 u = v16(initial_u);
                __m512 upl = v16(initial_upl);
                __m512 umi = v16(initial_umi);
                __m512 n = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(upl, upl), upl), umi);

                float3x16 direction = v3_normalize(v3_sub(
                    v3_add(v3_add(v3_mul(e1v, camera_ray.x), v3_mul(e2v, camera_ray.y)), v3_mul(e3v, camera_ray.z)),
                    e0v));
                if (use_depth_jitter) {
                    const __m512i depth_seed = pcg_hash_v16(_mm512_xor_si512(
                        pixel_x_v,
                        pcg_hash_v16(_mm512_xor_si512(
                            pixel_y_v, pcg_hash_v16(_mm512_xor_si512(_mm512_set1_epi32(sample),
                                                                    pcg_hash_v16(_mm512_set1_epi32(params.frames))))))));
                    __m512 depth_distance = _mm512_mul_ps(uint32_to_unit_float_v16(depth_seed),
                                                          _mm512_max_ps(_mm512_sub_ps(radius, v16(1.5f)), zero));
                    depth_distance = _mm512_div_ps(depth_distance, v16(10.0f));
                    depth_distance = _mm512_mul_ps(depth_distance, v16(params.step));
                    cam_pos = v3_add(cam_pos, v3_mul(direction, depth_distance));
                }

                float3x16 p = v3_mul(direction, n);
                const float3x16 p_init = p;
                const __m512 lz = _mm512_sub_ps(_mm512_mul_ps(cam_pos.x, p.y), _mm512_mul_ps(cam_pos.y, p.x));
                __m512 active = valid_mask;
                float4x16 accumulated = v4_zero();

                for (int step_index = 0; step_index < params.maxstep && any_lane(active); ++step_index) {
                    const float3x16 prev_pos = cam_pos;
                    const __m512 prev_dt = delta_t;

                    if (use_rk4) {
                        __m512 rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        __m512 gravity = _mm512_sub_ps(
                            zero,
                            _mm512_max_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf))));
                        __m512 uplsq = _mm512_mul_ps(upl, upl);
                        __m512 uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k11 = v3_mul(p, uu);
                        const float3x16 k12 = v3_mul(cam_pos, gravity);
                        const __m512 k_t1 = _mm512_div_ps(uplsq, _mm512_mul_ps(umi, umi));
                        const __m512 in_volume =
                            _mm512_and_ps(_mm512_and_ps(v16_cmp_ps(radius, v16(4.5f), _CMP_GT_OQ),
                                                        v16_cmp_ps(radius, v16(27.0f), _CMP_LT_OQ)),
                                          v16_cmp_ps(v16_abs(cam_pos.z), v16(3.0f), _CMP_LT_OQ));
                        __m512 disk_zone = _mm512_mul_ps(cam_pos.z, cam_pos.z);
                        disk_zone = _mm512_mul_ps(disk_zone, v16(0.25f));
                        disk_zone = _mm512_mul_ps(disk_zone, v16(0.15f));
                        disk_zone = _mm512_add_ps(v16(0.05f), disk_zone);
                        const __m512 zone_multiplier = v16_select(in_volume, disk_zone, one);
                        __m512 current_step =
                            _mm512_mul_ps(v16(params.step), v16_clamp(_mm512_sub_ps(radius, v16(0.54f)), 0.005f, 50.0f));
                        current_step = _mm512_mul_ps(current_step, zone_multiplier);
                        current_step = _mm512_mul_ps(current_step, v16(5.0f));
                        if (photon_ring_optimization) {
                            const __m512 distance = v16_abs(_mm512_sub_ps(radius, v16(1.866025f)));
                            current_step = _mm512_mul_ps(
                                current_step,
                                _mm512_add_ps(
                                    v16(0.05f),
                                    _mm512_mul_ps(v16(0.95f),
                                                  _mm512_div_ps(distance, _mm512_add_ps(distance, v16(0.12f))))));
                        }

                        __m512 step_half = _mm512_mul_ps(current_step, v16(0.5f));
                        float3x16 pos_tmp = v3_add(cam_pos, v3_mul(k11, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm512_div_ps(one, _mm512_mul_ps(v16(2.0f), radius));
                        upl = _mm512_add_ps(one, u);
                        umi = _mm512_sub_ps(one, u);
                        rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        gravity = _mm512_sub_ps(
                            zero,
                            _mm512_max_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm512_mul_ps(upl, upl);
                        uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k21 = v3_mul(v3_add(p, v3_mul(k12, step_half)), uu);
                        const float3x16 k22 = v3_mul(pos_tmp, gravity);
                        const __m512 k_t2 = _mm512_div_ps(uplsq, _mm512_mul_ps(umi, umi));

                        pos_tmp = v3_add(cam_pos, v3_mul(k21, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm512_div_ps(one, _mm512_mul_ps(v16(2.0f), radius));
                        upl = _mm512_add_ps(one, u);
                        umi = _mm512_sub_ps(one, u);
                        rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        gravity = _mm512_sub_ps(
                            zero,
                            _mm512_max_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm512_mul_ps(upl, upl);
                        uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k31 = v3_mul(v3_add(p, v3_mul(k22, step_half)), uu);
                        const float3x16 k32 = v3_mul(pos_tmp, gravity);
                        const __m512 k_t3 = _mm512_div_ps(uplsq, _mm512_mul_ps(umi, umi));

                        pos_tmp = v3_add(cam_pos, v3_mul(k31, current_step));
                        radius = v3_length(pos_tmp);
                        u = _mm512_div_ps(one, _mm512_mul_ps(v16(2.0f), radius));
                        upl = _mm512_add_ps(one, u);
                        umi = _mm512_sub_ps(one, u);
                        rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        gravity = _mm512_sub_ps(
                            zero,
                            _mm512_max_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf))));
                        uplsq = _mm512_mul_ps(upl, upl);
                        uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k41 = v3_mul(v3_add(p, v3_mul(k32, current_step)), uu);
                        const float3x16 k42 = v3_mul(pos_tmp, gravity);
                        const __m512 k_t4 = _mm512_div_ps(uplsq, _mm512_mul_ps(umi, umi));

                        step_half = _mm512_mul_ps(current_step, v16(0.16666666667f));
                        float3x16 position_sum = v3_add(k11, k41);
                        position_sum = v3_add(position_sum, v3_mul(k21, v16(2.0f)));
                        position_sum = v3_add(position_sum, v3_mul(k31, v16(2.0f)));
                        cam_pos = v3_add(cam_pos, v3_mul(position_sum, step_half));
                        float3x16 momentum_sum = v3_add(k12, k42);
                        momentum_sum = v3_add(momentum_sum, v3_mul(k22, v16(2.0f)));
                        momentum_sum = v3_add(momentum_sum, v3_mul(k32, v16(2.0f)));
                        p = v3_add(p, v3_mul(momentum_sum, step_half));
                        __m512 time_sum = _mm512_add_ps(k_t1, k_t4);
                        time_sum = _mm512_add_ps(time_sum, _mm512_mul_ps(v16(2.0f), k_t2));
                        time_sum = _mm512_add_ps(time_sum, _mm512_mul_ps(v16(2.0f), k_t3));
                        delta_t = _mm512_add_ps(delta_t, _mm512_mul_ps(step_half, time_sum));
                    } else {
                        __m512 rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        __m512 gravity =
                            _mm512_sub_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf)));
                        __m512 uplsq = _mm512_mul_ps(upl, upl);
                        __m512 uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k11 = v3_mul(p, uu);
                        const float3x16 k12 = v3_mul(cam_pos, gravity);
                        const __m512 in_volume =
                            _mm512_and_ps(_mm512_and_ps(v16_cmp_ps(radius, v16(4.5f), _CMP_GT_OQ),
                                                        v16_cmp_ps(radius, v16(37.0f), _CMP_LT_OQ)),
                                          v16_cmp_ps(v16_abs(cam_pos.z), v16(3.0f), _CMP_LT_OQ));
                        __m512 disk_zone = _mm512_mul_ps(cam_pos.z, cam_pos.z);
                        disk_zone = _mm512_mul_ps(disk_zone, v16(0.25f));
                        disk_zone = _mm512_mul_ps(disk_zone, v16(0.15f));
                        disk_zone = _mm512_add_ps(v16(0.05f), disk_zone);
                        const __m512 zone_multiplier = v16_select(in_volume, disk_zone, one);
                        __m512 current_step =
                            _mm512_mul_ps(v16(params.step), v16_clamp(_mm512_sub_ps(radius, v16(0.54f)), 0.005f, 50.0f));
                        current_step = _mm512_mul_ps(current_step, zone_multiplier);
                        if (photon_ring_optimization) {
                            const __m512 distance = v16_abs(_mm512_sub_ps(radius, v16(1.866025f)));
                            current_step = _mm512_mul_ps(
                                current_step,
                                _mm512_add_ps(
                                    v16(0.05f),
                                    _mm512_mul_ps(v16(0.95f),
                                                  _mm512_div_ps(distance, _mm512_add_ps(distance, v16(0.12f))))));
                        }
                        const __m512 step_half = _mm512_mul_ps(current_step, v16(0.5f));
                        const float3x16 pos_tmp = v3_add(cam_pos, v3_mul(k11, step_half));
                        radius = v3_length(pos_tmp);
                        u = _mm512_div_ps(one, _mm512_mul_ps(v16(2.0f), radius));
                        upl = _mm512_add_ps(one, u);
                        umi = _mm512_sub_ps(one, u);
                        rmhalf = _mm512_sub_ps(radius, v16(0.5f));
                        gravity =
                            _mm512_sub_ps(zero, _mm512_div_ps(_mm512_mul_ps(upl, _mm512_sub_ps(v16(2.0f), u)),
                                                              _mm512_mul_ps(_mm512_mul_ps(rmhalf, rmhalf), rmhalf)));
                        uplsq = _mm512_mul_ps(upl, upl);
                        uu = _mm512_div_ps(one, _mm512_mul_ps(uplsq, uplsq));
                        const float3x16 k21 = v3_mul(v3_add(p, v3_mul(k12, step_half)), uu);
                        const float3x16 k22 = v3_mul(pos_tmp, gravity);
                        const __m512 k_t2 = _mm512_div_ps(uplsq, _mm512_mul_ps(umi, umi));
                        cam_pos = v3_add(cam_pos, v3_mul(k21, current_step));
                        p = v3_add(p, v3_mul(k22, current_step));
                        delta_t = _mm512_add_ps(delta_t, _mm512_mul_ps(current_step, k_t2));
                    }

                    radius = v3_length(cam_pos);
                    u = _mm512_div_ps(one, _mm512_mul_ps(v16(2.0f), radius));
                    upl = _mm512_add_ps(one, u);
                    umi = _mm512_sub_ps(one, u);
                    n = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(upl, upl), upl), umi);
                    p = v3_mul(v3_normalize(p), n);

                    float3x16 disk_pos;
                    __m512 disk_time;
                    if (random_disk_sample) {
                        const __m512i random_seed = pcg_hash_v16(_mm512_xor_si512(
                            pixel_x_v,
                            pcg_hash_v16(_mm512_xor_si512(
                                pixel_y_v, pcg_hash_v16(_mm512_xor_si512(_mm512_set1_epi32(step_index),
                                                                        pcg_hash_v16(_mm512_set1_epi32(sample))))))));
                        const __m512 random_value = rand_float_v16(random_seed);
                        disk_pos =
                            v3_add(v3_mul(cam_pos, random_value), v3_mul(prev_pos, _mm512_sub_ps(one, random_value)));
                        disk_time = _mm512_add_ps(_mm512_mul_ps(delta_t, random_value),
                                                  _mm512_mul_ps(prev_dt, _mm512_sub_ps(one, random_value)));
                    } else {
                        disk_pos = v3_mul(v3_add(cam_pos, prev_pos), v16(0.5f));
                        disk_time = _mm512_mul_ps(_mm512_add_ps(prev_dt, delta_t), v16(0.5f));
                    }

                    const __m512 disk_radius_sq =
                        _mm512_add_ps(_mm512_mul_ps(disk_pos.x, disk_pos.x), _mm512_mul_ps(disk_pos.y, disk_pos.y));
                    __m512 in_disk =
                        _mm512_and_ps(_mm512_and_ps(v16_cmp_ps(disk_radius_sq, v16(24.4974f), _CMP_GT_OQ),
                                                    v16_cmp_ps(disk_radius_sq, v16(1225.0f), _CMP_LT_OQ)),
                                      v16_cmp_ps(v16_abs(cam_pos.z), v16(2.5f), _CMP_LT_OQ));
                    in_disk = _mm512_and_ps(in_disk, active);

                    if (any_lane(in_disk)) {

                        const __m512 disk_radius = _mm512_sqrt_ps(disk_radius_sq);
                        const __m512 sqrt_term = _mm512_sqrt_ps(
                            _mm512_sub_ps(_mm512_add_ps(v16(1.0f), _mm512_mul_ps(v16(4.0f), disk_radius_sq)),
                                          _mm512_mul_ps(v16(8.0f), disk_radius)));
                        const __m512 denominator = _mm512_add_ps(_mm512_mul_ps(v16(2.0f), disk_radius), one);
                        const __m512 td = _mm512_div_ps(denominator, sqrt_term);
                        const __m512 pd = _mm512_div_ps(
                            _mm512_mul_ps(_mm512_mul_ps(v16(8.0f), disk_radius), _mm512_sqrt_ps(disk_radius)),
                            _mm512_mul_ps(sqrt_term, _mm512_mul_ps(denominator, denominator)));
                        const __m512 rotation =
                            _mm512_div_ps(_mm512_mul_ps(pd, _mm512_sub_ps(v16(params.time), disk_time)), td);
                        const __m512 phi_final =
                            fast_mod2pi_v16(_mm512_add_ps(atan2_v16(disk_pos.y, disk_pos.x), rotation));
                        const float4x16 disk_parameters = sample_texture3d_v16(
                            disk, _mm512_mul_ps(phi_final, v16(0.15915494f)),
                            _mm512_add_ps(_mm512_div_ps(disk_pos.z, v16(5.0f)), v16(0.5f)),
                            _mm512_div_ps(_mm512_sub_ps(disk_radius, v16(4.9495f)), v16(30.0505f)), in_disk);
                        const __m512 p_init_dot_e0 = v3_dot(p_init, e0v);
                        __m512 g = _mm512_div_ps(_mm512_add_ps(v16(initial_factor * gamma), p_init_dot_e0),
                                                 _mm512_sub_ps(td, _mm512_mul_ps(pd, lz)));
                        g = v16_abs(g);
                        g = _mm512_add_ps(_mm512_sub_ps(g, one), one);
                        g = _mm512_max_ps(g, v16(0.01f));
                        const __m512 previous_length = v3_length(prev_pos);
                        const __m512 uuu =
                            _mm512_add_ps(one, _mm512_div_ps(one, _mm512_add_ps(previous_length, radius)));
                        const __m512 g2 = _mm512_mul_ps(g, g);
                        const __m512 g4 = _mm512_mul_ps(g2, g2);
                        const __m512 step_length = v3_length(v3_sub(cam_pos, prev_pos));
                        const __m512 kzg4 = _mm512_mul_ps(v16(2.0f), disk_parameters.z);
                        const __m512 extinction =
                            _mm512_sub_ps(one, exp_v16(_mm512_sub_ps(zero, _mm512_mul_ps(kzg4, kzg4))));
                        __m512 step_opacity = _mm512_mul_ps(
                            _mm512_mul_ps(_mm512_mul_ps(disk_parameters.x, opacity_change ? v16(3.0f) : v16(1.7f)),
                                          _mm512_mul_ps(uuu, uuu)),
                            _mm512_div_ps(_mm512_mul_ps(step_length, extinction), g));
                        if (opacity_change) {
                            const __m512 temp_eff = _mm512_mul_ps(disk_parameters.y, g);
                            const __m512 cold_factor = _mm512_add_ps(
                                one,
                                _mm512_mul_ps(v16(2.0f),
                                              v16_clamp(_mm512_div_ps(_mm512_sub_ps(v16(5000.0f), temp_eff), v16(3000.0f)),
                                                       0.0f, 1.0f)));
                            step_opacity = _mm512_mul_ps(step_opacity, cold_factor);
                        }
                        const __m512 opacity_weight = _mm512_sub_ps(one, exp_v16(_mm512_sub_ps(zero, step_opacity)));
                        float4x16 emission;
                        if (disk_doppler_follows_background) {
                            const float4x16 color = sample_texture2d_v16(
                                color_lut, _mm512_div_ps(_mm512_sub_ps(disk_parameters.y, v16(510.0f)), v16(20000.0f)),
                                v16(0.5f), in_disk);
                            const float3x16 shifted = rgb_three_line_frequency_shift_v16({color.x, color.y, color.z}, g);
                            const __m512 intensity = _mm512_mul_ps(disk_parameters.z, g4);
                            emission = {_mm512_mul_ps(shifted.x, intensity), _mm512_mul_ps(shifted.y, intensity),
                                        _mm512_mul_ps(shifted.z, intensity), one};
                        } else {
                            emission =
                                disk_emission_dep_v16(_mm512_max_ps(_mm512_mul_ps(disk_parameters.y, g), v16(1000.0f)),
                                                     _mm512_mul_ps(disk_parameters.z, g4), color_lut, in_disk);
                        }
                        const __m512 remaining = _mm512_sub_ps(one, accumulated.w);
                        const __m512 contribution = _mm512_mul_ps(remaining, opacity_weight);
                        float4x16 updated{_mm512_add_ps(accumulated.x, _mm512_mul_ps(emission.x, contribution)),
                                         _mm512_add_ps(accumulated.y, _mm512_mul_ps(emission.y, contribution)),
                                         _mm512_add_ps(accumulated.z, _mm512_mul_ps(emission.z, contribution)),
                                         _mm512_add_ps(accumulated.w, contribution)};
                        accumulated = v4_select(in_disk, updated, accumulated);
                    }

                    active = _mm512_and_ps(active,
                                           _mm512_and_ps(_mm512_and_ps(v16_cmp_ps(radius, v16(0.55f), _CMP_GE_OQ),
                                                                       v16_cmp_ps(radius, v16(140.0f), _CMP_LE_OQ)),
                                                         v16_cmp_ps(accumulated.w, v16(0.99f), _CMP_LE_OQ)));
                }

                const __m512 final_valid = _mm512_and_ps(valid_mask, v16_cmp_ps(radius, v16(0.55f), _CMP_GE_OQ));
                const float3x16 final_direction = v3_normalize(p);
                const __m512 phi = atan2_v16(final_direction.y, _mm512_sub_ps(zero, final_direction.x));
                const __m512 theta = asin_v16(_mm512_sub_ps(zero, final_direction.z));
                float4x16 background_color =
                    sample_texture2d_v16(background, _mm512_add_ps(_mm512_mul_ps(phi, v16(0.1591549f)), v16(0.5f)),
                                        _mm512_add_ps(_mm512_mul_ps(theta, v16(0.3183099f)), v16(0.5f)), final_valid);
                if (background_doppler) {
                    __m512 g_sky = v16_abs(_mm512_add_ps(v16(initial_factor * gamma), v3_dot(p_init, e0v)));
                    g_sky = v16_clamp(g_sky, 1e-4f, 20.0f);
                    const float3x16 shifted = rgb_three_line_frequency_shift_v16(
                        {background_color.x, background_color.y, background_color.z}, g_sky);
                    background_color.x = shifted.x;
                    background_color.y = shifted.y;
                    background_color.z = shifted.z;
                }
                const __m512 remaining = _mm512_sub_ps(one, accumulated.w);
                const float4x16 visible_color = v4_add(accumulated, v4_mul(background_color, remaining));
                const float4x16 captured_color{accumulated.x, accumulated.y, accumulated.z, one};
                buffer = v4_add(buffer, v4_select(final_valid, visible_color, captured_color));
            }

            buffer = v4_mul(buffer, v16(1.0f / static_cast<float>(params.jitternum)));
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
