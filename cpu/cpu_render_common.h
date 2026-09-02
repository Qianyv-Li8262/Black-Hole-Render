// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Qianyv-Li8262
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

enum RenderFlag : std::uint32_t {
    USE_RK4 = 1u << 0,
    DEPTH_JITTER = 1u << 1,
    RAND_SAMP_DISK = 1u << 2,
    OPACITY_CHANGE = 1u << 3,
    BACKGROUND_DOPPLER = 1u << 4,
    DISK_DOPPLER_FOLLOW_BACKGROUND = 1u << 5,
    PHOTON_RING_OPTIMIZATION = 1u << 6,
};

// Fields are exposed as cpu_render_native.RenderParams via pybind11.
struct render_params {
    float time;
    float cam_pos_x;
    float cam_pos_y;
    float cam_pos_z;
    float fwd_x;
    float fwd_y;
    float fwd_z;
    float right_x;
    float right_y;
    float right_z;
    float up_x;
    float up_y;
    float up_z;
    float vfwd;
    float vright;
    float vup;
    int imgwidth;
    int imgheight;
    float physwidth;
    float physheight;
    float focal_length;
    float step;
    int maxstep;
    int jitternum;
    int frames;
    std::uint32_t flags;
};

struct float2 {
    float x;
    float y;
};

struct float3 {
    float x;
    float y;
    float z;
};

struct float4 {
    float x;
    float y;
    float z;
    float w;
};

inline float2 make_float2(float x, float y)
{
    return {x, y};
}
inline float3 make_float3(float x, float y, float z)
{
    return {x, y, z};
}
inline float4 make_float4(float x, float y, float z, float w)
{
    return {x, y, z, w};
}

inline float rsqrtf_cpu(float value)
{
    return 1.0f / std::sqrt(value);
}

static inline float3 normalize(float3 value)
{
    const float inv_norm = rsqrtf_cpu(value.x * value.x + value.y * value.y + value.z * value.z);
    return make_float3(value.x * inv_norm, value.y * inv_norm, value.z * inv_norm);
}

static inline float length(float3 value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

static inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
static inline float3 operator*(float3 a, float scalar)
{
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}
static inline float3 operator*(float scalar, float3 a)
{
    return a * scalar;
}
static inline float4 operator*(float4 a, float scalar)
{
    return make_float4(a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar);
}
static inline float3 operator/(float3 a, float scalar)
{
    return make_float3(a.x / scalar, a.y / scalar, a.z / scalar);
}
static inline float operator*(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float saturate(float value)
{
    return std::fmin(std::fmax(value, 0.0f), 1.0f);
}
static inline bool has_flag(const render_params &params, RenderFlag flag)
{
    return (params.flags & flag) != 0;
}

// CUDA's normalized, linear, clamp-addressing texture behaviour.  CUDA's
// linear filter addresses the centre of texel 0 at coordinate 0.5 / extent.
struct Texture2D {
    const float *data;
    int width;
    int height;
    int channels;
    bool linear_filter;

    float4 texel(int x, int y) const
    {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        const float *value = data + (static_cast<std::size_t>(y) * width + x) * channels;
        return make_float4(value[0], value[1], value[2], channels == 4 ? value[3] : 0.0f);
    }

    float4 sample(float u, float v) const
    {
        if (!linear_filter)
            return texel(static_cast<int>(std::floor(u * width)), static_cast<int>(std::floor(v * height)));
        const float x = u * width - 0.5f;
        const float y = v * height - 0.5f;
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const float tx = x - x0;
        const float ty = y - y0;

        const float4 a = texel(x0, y0);
        const float4 b = texel(x0 + 1, y0);
        const float4 c = texel(x0, y0 + 1);
        const float4 d = texel(x0 + 1, y0 + 1);
        const float4 top = a * (1.0f - tx) + b * tx;
        const float4 bottom = c * (1.0f - tx) + d * tx;
        return top * (1.0f - ty) + bottom * ty;
    }
};

// The on-disk LUT has logical layout (radius, height, azimuth, channels),
// matching the NumPy array passed into create_texture_array_3d in cuda_tex.py.
struct Texture3D {
    const float *data;
    int radius;
    int height;
    int azimuth;
    int channels;
    bool wrap_azimuth;
    bool wrap_height;

    static int address_index(int value, int extent, bool wrap)
    {
        if (!wrap)
            return std::clamp(value, 0, extent - 1);
        value %= extent;
        return value < 0 ? value + extent : value;
    }

    float4 texel(int x, int y, int z) const
    {
        x = address_index(x, azimuth, wrap_azimuth);
        y = address_index(y, height, wrap_height);
        z = std::clamp(z, 0, radius - 1);
        const float *value = data + ((static_cast<std::size_t>(z) * height + y) * azimuth + x) * channels;
        return make_float4(value[0], value[1], value[2], channels == 4 ? value[3] : 0.0f);
    }

    float4 sample(float u, float v, float w) const
    {
        const float x = u * azimuth - 0.5f;
        const float y = v * height - 0.5f;
        const float z = w * radius - 0.5f;
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int z0 = static_cast<int>(std::floor(z));
        const float tx = x - x0;
        const float ty = y - y0;
        const float tz = z - z0;

        const float4 c000 = texel(x0, y0, z0);
        const float4 c100 = texel(x0 + 1, y0, z0);
        const float4 c010 = texel(x0, y0 + 1, z0);
        const float4 c110 = texel(x0 + 1, y0 + 1, z0);
        const float4 c001 = texel(x0, y0, z0 + 1);
        const float4 c101 = texel(x0 + 1, y0, z0 + 1);
        const float4 c011 = texel(x0, y0 + 1, z0 + 1);
        const float4 c111 = texel(x0 + 1, y0 + 1, z0 + 1);

        const float4 c00 = c000 * (1.0f - tx) + c100 * tx;
        const float4 c10 = c010 * (1.0f - tx) + c110 * tx;
        const float4 c01 = c001 * (1.0f - tx) + c101 * tx;
        const float4 c11 = c011 * (1.0f - tx) + c111 * tx;
        const float4 c0 = c00 * (1.0f - ty) + c10 * ty;
        const float4 c1 = c01 * (1.0f - ty) + c11 * ty;
        return c0 * (1.0f - tz) + c1 * tz;
    }
};

static inline float3 boost(float3 b, float3 vec, float gamma)
{
    const float len2 = b.x * b.x + b.y * b.y + b.z * b.z;
    if (len2 < 1e-12f)
        return vec;
    const float3 bnor = b * rsqrtf_cpu(len2);
    return vec + (gamma - 1.0f) * (bnor * vec) * bnor;
}

static inline std::uint32_t pcg_hash(std::uint32_t input)
{
    const std::uint32_t state = input * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static inline float rand_float(std::uint32_t seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4u;
    seed *= 0x27d4eb2du;
    seed ^= seed >> 15u;
    return static_cast<float>(seed) / 4294967296.0f;
}

static inline std::uint32_t reverse_bits(std::uint32_t value)
{
    value = ((value & 0x55555555u) << 1u) | ((value >> 1u) & 0x55555555u);
    value = ((value & 0x33333333u) << 2u) | ((value >> 2u) & 0x33333333u);
    value = ((value & 0x0f0f0f0fu) << 4u) | ((value >> 4u) & 0x0f0f0f0fu);
    value = ((value & 0x00ff00ffu) << 8u) | ((value >> 8u) & 0x00ff00ffu);
    return (value << 16u) | (value >> 16u);
}

static inline float2 hammersley(int index, int count, std::uint32_t pixel_x, std::uint32_t pixel_y,
                                std::uint32_t frames)
{
    const float h_x = static_cast<float>(index) / static_cast<float>(count);
    const float h_y = static_cast<float>(reverse_bits(static_cast<std::uint32_t>(index))) * 2.3283064365386963e-10f;
    const float shift_x = rand_float(pixel_x ^ pcg_hash(pixel_y) ^ (frames * 1919810u));
    const float shift_y = rand_float(pixel_y ^ pcg_hash(pixel_x * 114514u) ^ (frames * 1919810u));
    const float j_x = h_x + shift_x - std::floor(h_x + shift_x);
    const float j_y = h_y + shift_y - std::floor(h_y + shift_y);
    return make_float2(j_x, j_y);
}

static inline void tdpd(float radius, float *td, float *pd)
{
    const float radius_sq = radius * radius;
    const float sqrt_term = std::sqrt(1.0f + 4.0f * radius_sq - 8.0f * radius);
    const float denom = 2.0f * radius + 1.0f;
    *td = denom / sqrt_term;
    *pd = 8.0f * radius * std::sqrt(radius) / (sqrt_term * denom * denom);
}

static inline float fast_mod2pi(float value)
{
    return value - std::floor(value * 0.159154943f) * 6.283185307f;
}

static inline float smoothstepf(float edge0, float edge1, float value)
{
    const float t = saturate((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

static inline float3 wavelength_to_rgb(float lambda_nm)
{
    constexpr float UV_FADE0 = 330.0f;
    constexpr float UV_FADE1 = 410.0f;
    constexpr float IR_FADE0 = 720.0f;
    constexpr float IR_FADE1 = 860.0f;
    const float uv_to_visible = smoothstepf(UV_FADE0, UV_FADE1, lambda_nm);
    const float visible_to_ir = smoothstepf(IR_FADE0, IR_FADE1, lambda_nm);
    const float uv_weight = 1.0f - uv_to_visible;
    const float ir_weight = visible_to_ir;
    const float visible_weight = uv_to_visible * (1.0f - visible_to_ir);
    const float wavelength = std::fmin(std::fmax(lambda_nm, 380.0f), 780.0f);

    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    if (wavelength < 440.0f) {
        red = -(wavelength - 440.0f) / 60.0f;
        blue = 1.0f;
    } else if (wavelength < 490.0f) {
        green = (wavelength - 440.0f) / 50.0f;
        blue = 1.0f;
    } else if (wavelength < 510.0f) {
        green = 1.0f;
        blue = -(wavelength - 510.0f) / 20.0f;
    } else if (wavelength < 580.0f) {
        red = (wavelength - 510.0f) / 70.0f;
        green = 1.0f;
    } else if (wavelength < 645.0f) {
        red = 1.0f;
        green = -(wavelength - 645.0f) / 65.0f;
    } else {
        red = 1.0f;
    }

    float edge_factor = 1.0f;
    if (wavelength < 420.0f)
        edge_factor = 0.3f + 0.7f * (wavelength - 380.0f) / 40.0f;
    else if (wavelength > 700.0f)
        edge_factor = 0.3f + 0.7f * (780.0f - wavelength) / 80.0f;

    const float3 visible_rgb = make_float3(red, green, blue) * edge_factor;
    const float3 outside_uv = make_float3(0.55f, 0.0f, 1.0f) * 0.45f;
    const float3 outside_ir = make_float3(1.0f, 0.0f, 0.0f) * 0.45f;
    return visible_rgb * visible_weight + outside_uv * uv_weight + outside_ir * ir_weight;
}

static inline float3 rgb_three_line_frequency_shift(float3 rgb, float g)
{
    g = std::fmax(g, 1e-6f);
    const float3 c_red = wavelength_to_rgb(700.0f / g);
    const float3 c_green = wavelength_to_rgb(546.1f / g);
    const float3 c_blue = wavelength_to_rgb(435.8f / g);
    constexpr float green_leaks_to_red = (546.1f - 510.0f) / 70.0f;
    constexpr float blue_leaks_to_red = (440.0f - 435.8f) / 60.0f;
    const float3 coeff = make_float3(rgb.x - green_leaks_to_red * rgb.y - blue_leaks_to_red * rgb.z, rgb.y, rgb.z);
    float3 output = coeff.x * c_red + coeff.y * c_green + coeff.z * c_blue;
    const float g3 = g * g * g;
    output = output * g3;
    output.x = std::fmax(output.x, 0.0f);
    output.y = std::fmax(output.y, 0.0f);
    output.z = std::fmax(output.z, 0.0f);
    return output;
}

static inline float4 disk_emission_dep(float temperature, float intensity, const Texture2D &color_lut)
{
    const float4 color = color_lut.sample((temperature - 510.0f) / 20000.0f, 0.5f);
    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}

void render_one_tile_scalar(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                            const render_params &params, int tile_width, int tile_height, int tile_x, int tile_y);

bool cpu_supports_avx2();

bool cpu_supports_avx512();

void render_one_tile_avx2(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                          const render_params &params, int tile_width, int tile_height, int tile_x, int tile_y);

void render_one_tile_avx512(float *raw, const Texture2D &background, const Texture3D &disk,
                            const Texture2D &color_lut, const render_params &params, int tile_width,
                            int tile_height, int tile_x, int tile_y);
