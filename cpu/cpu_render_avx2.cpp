// Eight-ray AVX2 implementations. The accurate path keeps scalar texture and
// transcendental work; the experimental path vectorizes those operations too.

#include "cpu_render_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

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

static inline void store4(float4x8 value, float *x, float *y, float *z, float *w)
{
    _mm256_store_ps(x, value.x);
    _mm256_store_ps(y, value.y);
    _mm256_store_ps(z, value.z);
    _mm256_store_ps(w, value.w);
}

static inline float4x8 v4_sub(float4x8 a, float4x8 b)
{
    return {_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y), _mm256_sub_ps(a.z, b.z), _mm256_sub_ps(a.w, b.w)};
}

static inline float4x8 v4_select(__m256 mask, float4x8 when_true, float4x8 when_false)
{
    return {v8_select(mask, when_true.x, when_false.x), v8_select(mask, when_true.y, when_false.y),
            v8_select(mask, when_true.z, when_false.z), v8_select(mask, when_true.w, when_false.w)};
}

static inline __m256i i8_select(__m256i mask, __m256i when_true, __m256i when_false)
{
    return _mm256_or_si256(_mm256_and_si256(mask, when_true), _mm256_andnot_si256(mask, when_false));
}

static inline __m256i i8_clamp(__m256i value, int low, int high)
{
    return _mm256_min_epi32(_mm256_max_epi32(value, _mm256_set1_epi32(low)), _mm256_set1_epi32(high));
}

static inline __m256i i8_wrap_once(__m256i value, int extent)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i size = _mm256_set1_epi32(extent);
    value = i8_select(_mm256_cmpgt_epi32(zero, value), _mm256_add_epi32(value, size), value);
    return i8_select(_mm256_cmpgt_epi32(value, _mm256_set1_epi32(extent - 1)), _mm256_sub_epi32(value, size), value);
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

// Cephes-derived exp approximation with range reduction to powers of two.
// Its relative error is suitable for the opacity calculation and it stays
// entirely in eight AVX2 lanes.
static inline __m256 exp_v8(__m256 value)
{
    value = v8_clamp(value, -88.3762626647949f, 88.3762626647949f);
    __m256 exponent = _mm256_add_ps(_mm256_mul_ps(value, v8(1.44269504088896341f)), v8(0.5f));
    exponent = _mm256_floor_ps(exponent);
    value = _mm256_sub_ps(value, _mm256_mul_ps(exponent, v8(0.693359375f)));
    value = _mm256_sub_ps(value, _mm256_mul_ps(exponent, v8(-2.12194440e-4f)));
    const __m256 squared = _mm256_mul_ps(value, value);

    __m256 polynomial = v8(1.9875691500e-4f);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), v8(1.3981999507e-3f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), v8(8.3334519073e-3f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), v8(4.1665795894e-2f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), v8(1.6666665459e-1f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, value), v8(5.0000001201e-1f));
    polynomial = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(polynomial, squared), value), v8(1.0f));

    __m256i power = _mm256_cvttps_epi32(exponent);
    power = _mm256_slli_epi32(_mm256_add_epi32(power, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(power));
}

static inline __m256 atan_reduced_v8(__m256 value)
{
    const __m256 squared = _mm256_mul_ps(value, value);
    __m256 polynomial = v8(8.05374449538e-2f);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, squared), v8(-1.38776856032e-1f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, squared), v8(1.99777106478e-1f));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, squared), v8(-3.33329491539e-1f));
    return _mm256_add_ps(value, _mm256_mul_ps(_mm256_mul_ps(value, squared), polynomial));
}

static inline __m256 atan_unit_v8(__m256 value)
{
    const __m256 reduce = _mm256_cmp_ps(value, v8(0.4142135623730950f), _CMP_GT_OQ);
    const __m256 transformed = _mm256_div_ps(_mm256_sub_ps(value, v8(1.0f)), _mm256_add_ps(value, v8(1.0f)));
    return v8_select(reduce, _mm256_add_ps(v8(0.7853981633974483f), atan_reduced_v8(transformed)),
                     atan_reduced_v8(value));
}

static inline __m256 atan2_v8(__m256 y, __m256 x)
{
    const __m256 abs_x = v8_abs(x);
    const __m256 abs_y = v8_abs(y);
    const __m256 y_larger = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
    const __m256 maximum = _mm256_max_ps(abs_x, abs_y);
    const __m256 minimum = _mm256_min_ps(abs_x, abs_y);
    const __m256 nonzero = _mm256_cmp_ps(maximum, _mm256_setzero_ps(), _CMP_GT_OQ);
    const __m256 ratio = v8_select(nonzero, _mm256_div_ps(minimum, maximum), _mm256_setzero_ps());
    __m256 angle = atan_unit_v8(ratio);
    angle = v8_select(y_larger, _mm256_sub_ps(v8(1.5707963267948966f), angle), angle);
    angle = v8_select(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ), _mm256_sub_ps(v8(3.1415926535897932f), angle),
                      angle);
    return _mm256_xor_ps(angle, _mm256_and_ps(y, _mm256_set1_ps(-0.0f)));
}

static inline __m256 asin_v8(__m256 value)
{
    value = v8_clamp(value, -1.0f, 1.0f);
    const __m256 adjacent = _mm256_sqrt_ps(_mm256_max_ps(
        _mm256_setzero_ps(), _mm256_mul_ps(_mm256_sub_ps(v8(1.0f), value), _mm256_add_ps(v8(1.0f), value))));
    return atan2_v8(value, adjacent);
}

static inline __m256 fast_mod2pi_v8(__m256 value)
{
    return _mm256_sub_ps(value,
                         _mm256_mul_ps(_mm256_floor_ps(_mm256_mul_ps(value, v8(0.159154943f))), v8(6.283185307f)));
}

static inline float4x8 gather_texels(const float *data, __m256i texel_index, int channels, __m256 mask)
{
    const __m256i base = _mm256_mullo_epi32(texel_index, _mm256_set1_epi32(channels));
    const __m256 zero = _mm256_setzero_ps();
    const __m256 x = _mm256_mask_i32gather_ps(zero, data, base, mask, 4);
    const __m256 y = _mm256_mask_i32gather_ps(zero, data, _mm256_add_epi32(base, _mm256_set1_epi32(1)), mask, 4);
    const __m256 z = _mm256_mask_i32gather_ps(zero, data, _mm256_add_epi32(base, _mm256_set1_epi32(2)), mask, 4);
    const __m256 w = channels == 4
                         ? _mm256_mask_i32gather_ps(zero, data, _mm256_add_epi32(base, _mm256_set1_epi32(3)), mask, 4)
                         : zero;
    return {x, y, z, w};
}

static inline float4x8 lerp_v4(float4x8 a, float4x8 b, __m256 fraction)
{
    return v4_add(a, v4_mul(v4_sub(b, a), fraction));
}

static inline float4x8 sample_texture2d_v8(const Texture2D &texture, __m256 u, __m256 v, __m256 mask)
{
    const __m256 x = _mm256_sub_ps(_mm256_mul_ps(u, v8(static_cast<float>(texture.width))), v8(0.5f));
    const __m256 y = _mm256_sub_ps(_mm256_mul_ps(v, v8(static_cast<float>(texture.height))), v8(0.5f));
    const __m256 floor_x = _mm256_floor_ps(x);
    const __m256 floor_y = _mm256_floor_ps(y);
    const __m256 tx = _mm256_sub_ps(x, floor_x);
    const __m256 ty = _mm256_sub_ps(y, floor_y);
    const __m256i x0 = i8_clamp(_mm256_cvttps_epi32(floor_x), 0, texture.width - 1);
    const __m256i x1 =
        i8_clamp(_mm256_add_epi32(_mm256_cvttps_epi32(floor_x), _mm256_set1_epi32(1)), 0, texture.width - 1);
    const __m256i y0 = i8_clamp(_mm256_cvttps_epi32(floor_y), 0, texture.height - 1);
    const __m256i y1 =
        i8_clamp(_mm256_add_epi32(_mm256_cvttps_epi32(floor_y), _mm256_set1_epi32(1)), 0, texture.height - 1);
    const __m256i row0 = _mm256_mullo_epi32(y0, _mm256_set1_epi32(texture.width));
    const __m256i row1 = _mm256_mullo_epi32(y1, _mm256_set1_epi32(texture.width));
    const float4x8 top = lerp_v4(gather_texels(texture.data, _mm256_add_epi32(row0, x0), texture.channels, mask),
                                 gather_texels(texture.data, _mm256_add_epi32(row0, x1), texture.channels, mask), tx);
    const float4x8 bottom =
        lerp_v4(gather_texels(texture.data, _mm256_add_epi32(row1, x0), texture.channels, mask),
                gather_texels(texture.data, _mm256_add_epi32(row1, x1), texture.channels, mask), tx);
    return lerp_v4(top, bottom, ty);
}

static inline float4x8 sample_texture3d_v8(const Texture3D &texture, __m256 u, __m256 v, __m256 w, __m256 mask)
{
    const __m256 x = _mm256_sub_ps(_mm256_mul_ps(u, v8(static_cast<float>(texture.azimuth))), v8(0.5f));
    const __m256 y = _mm256_sub_ps(_mm256_mul_ps(v, v8(static_cast<float>(texture.height))), v8(0.5f));
    const __m256 z = _mm256_sub_ps(_mm256_mul_ps(w, v8(static_cast<float>(texture.radius))), v8(0.5f));
    const __m256 floor_x = _mm256_floor_ps(x);
    const __m256 floor_y = _mm256_floor_ps(y);
    const __m256 floor_z = _mm256_floor_ps(z);
    const __m256 tx = _mm256_sub_ps(x, floor_x);
    const __m256 ty = _mm256_sub_ps(y, floor_y);
    const __m256 tz = _mm256_sub_ps(z, floor_z);
    const __m256i raw_x0 = _mm256_cvttps_epi32(floor_x);
    const __m256i raw_y0 = _mm256_cvttps_epi32(floor_y);
    const __m256i raw_z0 = _mm256_cvttps_epi32(floor_z);
    const __m256i x0 = i8_wrap_once(raw_x0, texture.azimuth);
    const __m256i x1 = i8_wrap_once(_mm256_add_epi32(raw_x0, _mm256_set1_epi32(1)), texture.azimuth);
    const __m256i y0 = i8_wrap_once(raw_y0, texture.height);
    const __m256i y1 = i8_wrap_once(_mm256_add_epi32(raw_y0, _mm256_set1_epi32(1)), texture.height);
    const __m256i z0 = i8_clamp(raw_z0, 0, texture.radius - 1);
    const __m256i z1 = i8_clamp(_mm256_add_epi32(raw_z0, _mm256_set1_epi32(1)), 0, texture.radius - 1);
    const __m256i slice = _mm256_set1_epi32(texture.height * texture.azimuth);
    const __m256i row_width = _mm256_set1_epi32(texture.azimuth);
    const __m256i z0_base = _mm256_mullo_epi32(z0, slice);
    const __m256i z1_base = _mm256_mullo_epi32(z1, slice);
    const __m256i y0_base = _mm256_mullo_epi32(y0, row_width);
    const __m256i y1_base = _mm256_mullo_epi32(y1, row_width);

    const float4x8 c000 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z0_base, y0_base), x0), texture.channels, mask);
    const float4x8 c100 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z0_base, y0_base), x1), texture.channels, mask);
    const float4x8 c010 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z0_base, y1_base), x0), texture.channels, mask);
    const float4x8 c110 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z0_base, y1_base), x1), texture.channels, mask);
    const float4x8 c001 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z1_base, y0_base), x0), texture.channels, mask);
    const float4x8 c101 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z1_base, y0_base), x1), texture.channels, mask);
    const float4x8 c011 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z1_base, y1_base), x0), texture.channels, mask);
    const float4x8 c111 =
        gather_texels(texture.data, _mm256_add_epi32(_mm256_add_epi32(z1_base, y1_base), x1), texture.channels, mask);
    const float4x8 c00 = lerp_v4(c000, c100, tx);
    const float4x8 c10 = lerp_v4(c010, c110, tx);
    const float4x8 c01 = lerp_v4(c001, c101, tx);
    const float4x8 c11 = lerp_v4(c011, c111, tx);
    return lerp_v4(lerp_v4(c00, c10, ty), lerp_v4(c01, c11, ty), tz);
}

static inline __m256 smoothstep_v8(float edge0, float edge1, __m256 value)
{
    const __m256 t = v8_clamp(_mm256_div_ps(_mm256_sub_ps(value, v8(edge0)), v8(edge1 - edge0)), 0.0f, 1.0f);
    return _mm256_mul_ps(_mm256_mul_ps(t, t), _mm256_sub_ps(v8(3.0f), _mm256_mul_ps(v8(2.0f), t)));
}

static inline float3x8 wavelength_to_rgb_v8(__m256 lambda_nm)
{
    const __m256 uv_to_visible = smoothstep_v8(330.0f, 410.0f, lambda_nm);
    const __m256 visible_to_ir = smoothstep_v8(720.0f, 860.0f, lambda_nm);
    const __m256 uv_weight = _mm256_sub_ps(v8(1.0f), uv_to_visible);
    const __m256 ir_weight = visible_to_ir;
    const __m256 visible_weight = _mm256_mul_ps(uv_to_visible, _mm256_sub_ps(v8(1.0f), visible_to_ir));
    const __m256 wavelength = v8_clamp(lambda_nm, 380.0f, 780.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = v8(1.0f);
    __m256 red = zero;
    __m256 green = zero;
    __m256 blue = zero;

    const __m256 range0 = _mm256_cmp_ps(wavelength, v8(440.0f), _CMP_LT_OQ);
    const __m256 range1 = _mm256_andnot_ps(range0, _mm256_cmp_ps(wavelength, v8(490.0f), _CMP_LT_OQ));
    const __m256 below510 = _mm256_cmp_ps(wavelength, v8(510.0f), _CMP_LT_OQ);
    const __m256 range2 = _mm256_andnot_ps(_mm256_or_ps(range0, range1), below510);
    const __m256 below580 = _mm256_cmp_ps(wavelength, v8(580.0f), _CMP_LT_OQ);
    const __m256 range3 = _mm256_andnot_ps(_mm256_or_ps(_mm256_or_ps(range0, range1), range2), below580);
    const __m256 below645 = _mm256_cmp_ps(wavelength, v8(645.0f), _CMP_LT_OQ);
    const __m256 prior3 = _mm256_or_ps(_mm256_or_ps(range0, range1), _mm256_or_ps(range2, range3));
    const __m256 range4 = _mm256_andnot_ps(prior3, below645);
    const __m256 range5 = _mm256_andnot_ps(_mm256_or_ps(prior3, range4), _mm256_castsi256_ps(_mm256_set1_epi32(-1)));

    red = v8_select(range0, _mm256_div_ps(_mm256_sub_ps(v8(440.0f), wavelength), v8(60.0f)), red);
    blue = v8_select(range0, one, blue);
    green = v8_select(range1, _mm256_div_ps(_mm256_sub_ps(wavelength, v8(440.0f)), v8(50.0f)), green);
    blue = v8_select(range1, one, blue);
    green = v8_select(range2, one, green);
    blue = v8_select(range2, _mm256_div_ps(_mm256_sub_ps(v8(510.0f), wavelength), v8(20.0f)), blue);
    red = v8_select(range3, _mm256_div_ps(_mm256_sub_ps(wavelength, v8(510.0f)), v8(70.0f)), red);
    green = v8_select(range3, one, green);
    red = v8_select(range4, one, red);
    green = v8_select(range4, _mm256_div_ps(_mm256_sub_ps(v8(645.0f), wavelength), v8(65.0f)), green);
    red = v8_select(range5, one, red);

    __m256 edge_factor = one;
    const __m256 violet_edge = _mm256_add_ps(
        v8(0.3f), _mm256_mul_ps(v8(0.7f), _mm256_div_ps(_mm256_sub_ps(wavelength, v8(380.0f)), v8(40.0f))));
    const __m256 red_edge = _mm256_add_ps(
        v8(0.3f), _mm256_mul_ps(v8(0.7f), _mm256_div_ps(_mm256_sub_ps(v8(780.0f), wavelength), v8(80.0f))));
    edge_factor = v8_select(_mm256_cmp_ps(wavelength, v8(420.0f), _CMP_LT_OQ), violet_edge, edge_factor);
    edge_factor = v8_select(_mm256_cmp_ps(wavelength, v8(700.0f), _CMP_GT_OQ), red_edge, edge_factor);

    const float3x8 visible_rgb{_mm256_mul_ps(red, edge_factor), _mm256_mul_ps(green, edge_factor),
                               _mm256_mul_ps(blue, edge_factor)};
    const float3x8 outside_uv{v8(0.2475f), zero, v8(0.45f)};
    const float3x8 outside_ir{v8(0.45f), zero, zero};
    return v3_add(v3_add(v3_mul(visible_rgb, visible_weight), v3_mul(outside_uv, uv_weight)),
                  v3_mul(outside_ir, ir_weight));
}

static inline float3x8 rgb_three_line_frequency_shift_v8(float3x8 rgb, __m256 g)
{
    g = _mm256_max_ps(g, v8(1e-6f));
    const float3x8 c_red = wavelength_to_rgb_v8(_mm256_div_ps(v8(700.0f), g));
    const float3x8 c_green = wavelength_to_rgb_v8(_mm256_div_ps(v8(546.1f), g));
    const float3x8 c_blue = wavelength_to_rgb_v8(_mm256_div_ps(v8(435.8f), g));
    constexpr float green_leaks_to_red = (546.1f - 510.0f) / 70.0f;
    constexpr float blue_leaks_to_red = (440.0f - 435.8f) / 60.0f;
    const float3x8 coefficients{_mm256_sub_ps(_mm256_sub_ps(rgb.x, _mm256_mul_ps(v8(green_leaks_to_red), rgb.y)),
                                              _mm256_mul_ps(v8(blue_leaks_to_red), rgb.z)),
                                rgb.y, rgb.z};
    float3x8 output =
        v3_add(v3_add(v3_mul(c_red, coefficients.x), v3_mul(c_green, coefficients.y)), v3_mul(c_blue, coefficients.z));
    const __m256 g3 = _mm256_mul_ps(_mm256_mul_ps(g, g), g);
    output = v3_mul(output, g3);
    output.x = _mm256_max_ps(output.x, _mm256_setzero_ps());
    output.y = _mm256_max_ps(output.y, _mm256_setzero_ps());
    output.z = _mm256_max_ps(output.z, _mm256_setzero_ps());
    return output;
}

static inline float4x8 disk_emission_dep_v8(__m256 temperature, __m256 intensity, const Texture2D &color_lut,
                                            __m256 mask)
{
    const float4x8 color = sample_texture2d_v8(
        color_lut, _mm256_div_ps(_mm256_sub_ps(temperature, v8(510.0f)), v8(20000.0f)), v8(0.5f), mask);
    return {_mm256_mul_ps(color.x, intensity), _mm256_mul_ps(color.y, intensity), _mm256_mul_ps(color.z, intensity),
            v8(1.0f)};
}

bool cpu_supports_avx2()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 1);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    if (!osxsave || !avx || (_xgetbv(0) & 0x6) != 0x6)
        return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
#else
    return false;
#endif
}

bool cpu_supports_avx512()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 1);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    if (!osxsave || !avx)
        return false;
    // XMM, YMM, opmask, upper ZMM, and ZMM16-31 state must all be managed by the OS.
    if ((_xgetbv(0) & 0xe6) != 0xe6)
        return false;
    __cpuidex(registers, 7, 0);
    constexpr int avx512f_bit = 1 << 16;
    constexpr int avx512dq_bit = 1 << 17;
    return (registers[1] & avx512f_bit) != 0 && (registers[1] & avx512dq_bit) != 0;
#else
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#endif
#else
    return false;
#endif
}

void render_one_tile_avx2(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
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

                        const __m256 disk_radius = _mm256_sqrt_ps(disk_radius_sq);
                        const __m256 sqrt_term = _mm256_sqrt_ps(
                            _mm256_sub_ps(_mm256_add_ps(v8(1.0f), _mm256_mul_ps(v8(4.0f), disk_radius_sq)),
                                          _mm256_mul_ps(v8(8.0f), disk_radius)));
                        const __m256 denominator = _mm256_add_ps(_mm256_mul_ps(v8(2.0f), disk_radius), one);
                        const __m256 td = _mm256_div_ps(denominator, sqrt_term);
                        const __m256 pd = _mm256_div_ps(
                            _mm256_mul_ps(_mm256_mul_ps(v8(8.0f), disk_radius), _mm256_sqrt_ps(disk_radius)),
                            _mm256_mul_ps(sqrt_term, _mm256_mul_ps(denominator, denominator)));
                        const __m256 rotation =
                            _mm256_div_ps(_mm256_mul_ps(pd, _mm256_sub_ps(v8(params.time), disk_time)), td);
                        const __m256 phi_final =
                            fast_mod2pi_v8(_mm256_add_ps(atan2_v8(disk_pos.y, disk_pos.x), rotation));
                        const float4x8 disk_parameters = sample_texture3d_v8(
                            disk, _mm256_mul_ps(phi_final, v8(0.15915494f)),
                            _mm256_add_ps(_mm256_div_ps(disk_pos.z, v8(5.0f)), v8(0.5f)),
                            _mm256_div_ps(_mm256_sub_ps(disk_radius, v8(4.9495f)), v8(30.0505f)), in_disk);
                        const __m256 p_init_dot_e0 = v3_dot(p_init, e0v);
                        __m256 g = _mm256_div_ps(_mm256_add_ps(v8(initial_factor * gamma), p_init_dot_e0),
                                                 _mm256_sub_ps(td, _mm256_mul_ps(pd, lz)));
                        g = v8_abs(g);
                        g = _mm256_add_ps(_mm256_sub_ps(g, one), one);
                        g = _mm256_max_ps(g, v8(0.01f));
                        const __m256 previous_length = v3_length(prev_pos);
                        const __m256 uuu =
                            _mm256_add_ps(one, _mm256_div_ps(one, _mm256_add_ps(previous_length, radius)));
                        const __m256 g2 = _mm256_mul_ps(g, g);
                        const __m256 g4 = _mm256_mul_ps(g2, g2);
                        const __m256 step_length = v3_length(v3_sub(cam_pos, prev_pos));
                        const __m256 kzg4 = _mm256_mul_ps(v8(2.0f), disk_parameters.z);
                        const __m256 extinction =
                            _mm256_sub_ps(one, exp_v8(_mm256_sub_ps(zero, _mm256_mul_ps(kzg4, kzg4))));
                        __m256 step_opacity = _mm256_mul_ps(
                            _mm256_mul_ps(_mm256_mul_ps(disk_parameters.x, opacity_change ? v8(3.0f) : v8(1.7f)),
                                          _mm256_mul_ps(uuu, uuu)),
                            _mm256_div_ps(_mm256_mul_ps(step_length, extinction), g));
                        if (opacity_change) {
                            const __m256 temp_eff = _mm256_mul_ps(disk_parameters.y, g);
                            const __m256 cold_factor = _mm256_add_ps(
                                one,
                                _mm256_mul_ps(v8(2.0f),
                                              v8_clamp(_mm256_div_ps(_mm256_sub_ps(v8(5000.0f), temp_eff), v8(3000.0f)),
                                                       0.0f, 1.0f)));
                            step_opacity = _mm256_mul_ps(step_opacity, cold_factor);
                        }
                        const __m256 opacity_weight = _mm256_sub_ps(one, exp_v8(_mm256_sub_ps(zero, step_opacity)));
                        float4x8 emission;
                        if (disk_doppler_follows_background) {
                            const float4x8 color = sample_texture2d_v8(
                                color_lut, _mm256_div_ps(_mm256_sub_ps(disk_parameters.y, v8(510.0f)), v8(20000.0f)),
                                v8(0.5f), in_disk);
                            const float3x8 shifted = rgb_three_line_frequency_shift_v8({color.x, color.y, color.z}, g);
                            const __m256 intensity = _mm256_mul_ps(disk_parameters.z, g4);
                            emission = {_mm256_mul_ps(shifted.x, intensity), _mm256_mul_ps(shifted.y, intensity),
                                        _mm256_mul_ps(shifted.z, intensity), one};
                        } else {
                            emission =
                                disk_emission_dep_v8(_mm256_max_ps(_mm256_mul_ps(disk_parameters.y, g), v8(1000.0f)),
                                                     _mm256_mul_ps(disk_parameters.z, g4), color_lut, in_disk);
                        }
                        const __m256 remaining = _mm256_sub_ps(one, accumulated.w);
                        const __m256 contribution = _mm256_mul_ps(remaining, opacity_weight);
                        float4x8 updated{_mm256_add_ps(accumulated.x, _mm256_mul_ps(emission.x, contribution)),
                                         _mm256_add_ps(accumulated.y, _mm256_mul_ps(emission.y, contribution)),
                                         _mm256_add_ps(accumulated.z, _mm256_mul_ps(emission.z, contribution)),
                                         _mm256_add_ps(accumulated.w, contribution)};
                        accumulated = v4_select(in_disk, updated, accumulated);
                    }

                    active = _mm256_and_ps(active,
                                           _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(radius, v8(0.55f), _CMP_GE_OQ),
                                                                       _mm256_cmp_ps(radius, v8(140.0f), _CMP_LE_OQ)),
                                                         _mm256_cmp_ps(accumulated.w, v8(0.99f), _CMP_LE_OQ)));
                }

                const __m256 final_valid = _mm256_and_ps(valid_mask, _mm256_cmp_ps(radius, v8(0.55f), _CMP_GE_OQ));
                const float3x8 final_direction = v3_normalize(p);
                const __m256 phi = atan2_v8(final_direction.y, _mm256_sub_ps(zero, final_direction.x));
                const __m256 theta = asin_v8(_mm256_sub_ps(zero, final_direction.z));
                float4x8 background_color =
                    sample_texture2d_v8(background, _mm256_add_ps(_mm256_mul_ps(phi, v8(0.1591549f)), v8(0.5f)),
                                        _mm256_add_ps(_mm256_mul_ps(theta, v8(0.3183099f)), v8(0.5f)), final_valid);
                if (background_doppler) {
                    __m256 g_sky = v8_abs(_mm256_add_ps(v8(initial_factor * gamma), v3_dot(p_init, e0v)));
                    g_sky = v8_clamp(g_sky, 1e-4f, 20.0f);
                    const float3x8 shifted = rgb_three_line_frequency_shift_v8(
                        {background_color.x, background_color.y, background_color.z}, g_sky);
                    background_color.x = shifted.x;
                    background_color.y = shifted.y;
                    background_color.z = shifted.z;
                }
                const __m256 remaining = _mm256_sub_ps(one, accumulated.w);
                const float4x8 visible_color = v4_add(accumulated, v4_mul(background_color, remaining));
                const float4x8 captured_color{accumulated.x, accumulated.y, accumulated.z, one};
                buffer = v4_add(buffer, v4_select(final_valid, visible_color, captured_color));
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
