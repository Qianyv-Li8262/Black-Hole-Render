// CPU implementation of blackholekernel3_prebaked copy.cu.
//
// This file is built as a pybind11 extension.  The public functions accept
// NumPy arrays directly, avoiding the duplicated ctypes ABI declarations that
// used to live in cpu_render.py.

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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

static void render_one_time_impl(float *raw, const Texture2D &background, const Texture3D &disk,
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
    const float3 beta_global = beta.x * fwd + beta.y * right + beta.z * up;
    const float3 e0 = beta_global * gamma / (initial_upl * initial_upl);
    float3 e1_fwd = boost(beta, make_float3(1.0f, 0.0f, 0.0f), gamma);
    e1_fwd = (e1_fwd.x * fwd + e1_fwd.y * right + e1_fwd.z * up) / (initial_upl * initial_upl);
    float3 e2_right = boost(beta, make_float3(0.0f, 1.0f, 0.0f), gamma);
    e2_right = (e2_right.x * fwd + e2_right.y * right + e2_right.z * up) / (initial_upl * initial_upl);
    float3 e3_up = boost(beta, make_float3(0.0f, 0.0f, 1.0f), gamma);
    e3_up = (e3_up.x * fwd + e3_up.y * right + e3_up.z * up) / (initial_upl * initial_upl);

    for (int pixel_y = y_begin; pixel_y < y_end; ++pixel_y) {
        for (int pixel_x = x_begin; pixel_x < x_end; ++pixel_x) {
            float4 buffer = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            for (int sample = 0; sample < params.jitternum; ++sample) {
                // The CUDA source currently passes 1 rather than `frames` here.
                const float2 jitter = hammersley(sample, params.jitternum, static_cast<std::uint32_t>(pixel_x),
                                                 static_cast<std::uint32_t>(pixel_y), 1u);
                const float physical_x =
                    ((static_cast<float>(pixel_x) + jitter.x) / static_cast<float>(params.imgwidth) - 0.5f) *
                    params.physwidth;
                const float physical_y =
                    ((static_cast<float>(pixel_y) + jitter.y) / static_cast<float>(params.imgheight) - 0.5f) *
                    params.physheight;
                float3 cam_pos = initial_cam_pos;
                float delta_t = 0.0f;
                const float3 camera_ray = normalize(make_float3(params.focal_length, physical_x, -physical_y));

                float radius = length(cam_pos);
                float u = 1.0f / (2.0f * radius);
                float upl = 1.0f + u;
                float umi = 1.0f - u;
                float factor = upl / umi;
                float n = upl * upl * upl / umi;
                float3 direction =
                    normalize(camera_ray.x * e1_fwd + camera_ray.y * e2_right + camera_ray.z * e3_up - e0);
                if (has_flag(params, DEPTH_JITTER)) {
                    const std::uint32_t depth_seed =
                        pcg_hash(static_cast<std::uint32_t>(pixel_x) ^
                                 pcg_hash(static_cast<std::uint32_t>(pixel_y) ^
                                          pcg_hash(static_cast<std::uint32_t>(sample) ^
                                                   pcg_hash(static_cast<std::uint32_t>(params.frames)))));
                    const float depth_jitter = static_cast<float>(depth_seed) / 4294967296.0f;
                    cam_pos =
                        cam_pos + direction * (depth_jitter * std::fmax(radius - 1.5f, 0.0f) / 10.0f * params.step);
                }
                float3 p = direction * n;
                const float3 p_init = p;
                const float lz = cam_pos.x * p.y - cam_pos.y * p.x;
                bool trace = true;
                float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                for (int step_index = 0; step_index < params.maxstep && trace; ++step_index) {
                    const float3 prev_pos = cam_pos;
                    const float prev_dt = delta_t;

                    if (has_flag(params, USE_RK4)) {
                        float rmhalf = radius - 0.5f;
                        float gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        float uplsq = upl * upl;
                        float uu = 1.0f / (uplsq * uplsq);
                        const float3 k11 = p * uu;
                        const float3 k12 = gravity * cam_pos;
                        const float k_t1 = uplsq / (umi * umi);
                        const bool in_disk_volume = radius > 4.5f && radius < 27.0f && std::fabs(cam_pos.z) < 3.0f;
                        const float zone_multiplier =
                            in_disk_volume ? 0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f) : 1.0f;
                        float current_step =
                            params.step * std::fmin(50.0f, std::fmax(0.005f, radius - 0.54f)) * zone_multiplier * 5.0f;
                        if (has_flag(params, PHOTON_RING_OPTIMIZATION)) {
                            const float dist_to_ps = std::fabs(radius - 1.866025f);
                            current_step *= 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
                        }

                        float step_half = current_step * 0.5f;
                        float3 pos_tmp = cam_pos + step_half * k11;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k21 = (p + step_half * k12) * uu;
                        const float3 k22 = pos_tmp * gravity;
                        const float k_t2 = uplsq / (umi * umi);

                        pos_tmp = cam_pos + step_half * k21;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k31 = (p + step_half * k22) * uu;
                        const float3 k32 = pos_tmp * gravity;
                        const float k_t3 = uplsq / (umi * umi);

                        pos_tmp = cam_pos + current_step * k31;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -std::fmax(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k41 = (p + current_step * k32) * uu;
                        const float3 k42 = pos_tmp * gravity;
                        const float k_t4 = uplsq / (umi * umi);

                        step_half = current_step * 0.16666666667f;
                        cam_pos = cam_pos + step_half * (k11 + k41 + 2.0f * k21 + 2.0f * k31);
                        p = p + step_half * (k12 + k42 + 2.0f * k22 + 2.0f * k32);
                        delta_t += step_half * (k_t1 + k_t4 + 2.0f * k_t2 + 2.0f * k_t3);
                        radius = length(cam_pos);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        n = upl * upl * upl / umi;
                        p = normalize(p) * n;
                    } else {
                        float rmhalf = radius - 0.5f;
                        float gravity = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
                        float uplsq = upl * upl;
                        float uu = 1.0f / (uplsq * uplsq);
                        const float3 k11 = p * uu;
                        const float3 k12 = gravity * cam_pos;
                        const bool in_disk_volume = radius > 4.5f && radius < 37.0f && std::fabs(cam_pos.z) < 3.0f;
                        const float zone_multiplier =
                            in_disk_volume ? 0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f) : 1.0f;
                        float current_step =
                            params.step * std::fmin(50.0f, std::fmax(0.005f, radius - 0.54f)) * zone_multiplier;
                        if (has_flag(params, PHOTON_RING_OPTIMIZATION)) {
                            const float dist_to_ps = std::fabs(radius - 1.866025f);
                            current_step *= 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
                        }
                        const float step_half = current_step * 0.5f;
                        const float3 pos_tmp = cam_pos + step_half * k11;
                        radius = length(pos_tmp);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        rmhalf = radius - 0.5f;
                        gravity = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
                        uplsq = upl * upl;
                        uu = 1.0f / (uplsq * uplsq);
                        const float3 k21 = (p + step_half * k12) * uu;
                        const float3 k22 = pos_tmp * gravity;
                        const float k_t2 = uplsq / (umi * umi);
                        cam_pos = cam_pos + current_step * k21;
                        p = p + current_step * k22;
                        delta_t += current_step * k_t2;
                        radius = length(cam_pos);
                        u = 1.0f / (2.0f * radius);
                        upl = 1.0f + u;
                        umi = 1.0f - u;
                        n = upl * upl * upl / umi;
                        p = normalize(p) * n;
                    }

                    float3 disk_pos;
                    float disk_time;
                    if (has_flag(params, RAND_SAMP_DISK)) {
                        const float random_sample =
                            rand_float(pcg_hash(static_cast<std::uint32_t>(pixel_x) ^
                                                pcg_hash(static_cast<std::uint32_t>(pixel_y) ^
                                                         pcg_hash(static_cast<std::uint32_t>(step_index) ^
                                                                  pcg_hash(static_cast<std::uint32_t>(sample))))));
                        disk_pos = cam_pos * random_sample + prev_pos * (1.0f - random_sample);
                        disk_time = delta_t * random_sample + prev_dt * (1.0f - random_sample);
                    } else {
                        disk_pos = (cam_pos + prev_pos) * 0.5f;
                        disk_time = (prev_dt + delta_t) * 0.5f;
                    }

                    const float disk_radius_sq = disk_pos.x * disk_pos.x + disk_pos.y * disk_pos.y;
                    const bool in_disk =
                        disk_radius_sq > 24.4974f && disk_radius_sq < 1225.0f && std::fabs(cam_pos.z) < 2.5f;
                    if (in_disk) {
                        const float disk_radius = std::sqrt(disk_radius_sq);
                        float td = 0.0f;
                        float pd = 0.0f;
                        tdpd(disk_radius, &td, &pd);
                        const float rotation = pd * (params.time - disk_time) / td;
                        const float phi_final = fast_mod2pi(std::atan2(disk_pos.y, disk_pos.x) + rotation);
                        const float4 disk_parameters =
                            disk.sample(phi_final * 0.15915494f, (disk_pos.z / 2.5f) / 2.0f + 0.5f,
                                        (disk_radius - 4.9495f) / 30.0505f);
                        const float g = std::fmax(
                            (std::fabs((factor * gamma + p_init * e0) / (td - pd * lz)) - 1.0f) + 1.0f, 0.01f);
                        const float ravg2 = length(prev_pos) + radius;
                        const float uuu = 1.0f + 1.0f / ravg2;
                        const float g4 = g * g * g * g;
                        const float step_len = length(cam_pos - prev_pos);
                        const float kzg4 = 2.0f * disk_parameters.z;
                        float step_opacity;
                        if (has_flag(params, OPACITY_CHANGE)) {
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
                        if (has_flag(params, DISK_DOPPLER_FOLLOW_BACKGROUND)) {
                            float4 color = color_lut.sample((disk_parameters.y - 510.0f) / 20000.0f, 0.5f);
                            const float3 shifted_rgb =
                                rgb_three_line_frequency_shift(make_float3(color.x, color.y, color.z), g);
                            emission = make_float4(shifted_rgb.x * disk_parameters.z * g4,
                                                   shifted_rgb.y * disk_parameters.z * g4,
                                                   shifted_rgb.z * disk_parameters.z * g4, 1.0f);
                        } else {
                            emission = disk_emission_dep(std::fmax(disk_parameters.y * g, 1000.0f),
                                                         disk_parameters.z * g4, color_lut);
                        }

                        float temp_calc = std::fma(emission.x, -accumulated_color.w, emission.x);
                        accumulated_color.x += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = std::fma(emission.y, -accumulated_color.w, emission.y);
                        accumulated_color.y += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = std::fma(emission.z, -accumulated_color.w, emission.z);
                        accumulated_color.z += std::fma(temp_calc, temp_exp, temp_calc);
                        temp_calc = 1.0f - accumulated_color.w;
                        accumulated_color.w += std::fma(temp_calc, temp_exp, temp_calc);
                        if (accumulated_color.w > 0.99f)
                            trace = false;
                    }

                    if (radius < 0.55f || radius > 140.0f)
                        trace = false;
                }

                float4 color;
                if (radius >= 0.55f && !std::isnan(radius)) {
                    const float3 final_dir = normalize(p);
                    const float phi = std::atan2(final_dir.y, -final_dir.x);
                    const float theta = std::asin(-final_dir.z);
                    float4 bkgd = background.sample(phi * 0.1591549f + 0.5f, theta * 0.3183099f + 0.5f);
                    if (has_flag(params, BACKGROUND_DOPPLER)) {
                        float g_sky = std::fabs(factor * gamma + p_init * e0);
                        g_sky = std::fmin(std::fmax(g_sky, 1e-4f), 20.0f);
                        const float3 shifted_rgb =
                            rgb_three_line_frequency_shift(make_float3(bkgd.x, bkgd.y, bkgd.z), g_sky);
                        bkgd.x = shifted_rgb.x;
                        bkgd.y = shifted_rgb.y;
                        bkgd.z = shifted_rgb.z;
                    }
                    color = accumulated_color + bkgd * (1.0f - accumulated_color.w);
                } else {
                    color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
                }
                buffer = buffer + color;
            }

            buffer = buffer * (1.0f / static_cast<float>(params.jitternum));
            float *output = raw + (static_cast<std::size_t>(pixel_y) * params.imgwidth + pixel_x) * 4;
            output[0] = buffer.x;
            output[1] = buffer.y;
            output[2] = buffer.z;
            output[3] = 0.0f;
        }
    }
}

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
            // matching extractBright in postprocess_downup copy.cu.
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

static int render(float *raw, const Texture2D &background, const Texture3D &disk, const Texture2D &color_lut,
                  const render_params &params, int tile_width, int tile_height, int requested_workers)
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
                render_one_time_impl(raw, background, disk, color_lut, params, tile_width, tile_height,
                                     tile_index % tiles_x, tile_index / tiles_x);
            }
        });
    }
    for (std::thread &worker : workers)
        worker.join();
    return 0;
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

template <typename T>
static T *writable_image_from_numpy(py::array &array, const char *name, int &width, int &height)
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
        status = render(raw, background_texture, disk_texture, color_lut_texture, params, tile_width, tile_height,
                        workers);
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
    py::array_t<float> output(py::array::ShapeContainer{
        static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(4)});
    render_into_numpy(output, background, prebaked_disk, color_lut, params, tile_width, tile_height, workers);
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

PYBIND11_MODULE(cpu_render_native, module)
{
    module.doc() = "CPU black-hole renderer with zero-copy NumPy array inputs and outputs.";

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

    module.def("render_into", &render_into_numpy, py::arg("raw_img"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Render directly into a C-contiguous float32 (height, width, 4) NumPy array.");
    module.def("render_new", &render_new_numpy, py::arg("width"), py::arg("height"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32,
               py::arg("tile_height") = 8, py::arg("workers") = 0,
               "Allocate and return a float32 (height, width, 4) NumPy image.");
    module.def("render_rgba8_into", &render_rgba8_into_numpy, py::arg("output_img"), py::arg("background"),
               py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"), py::arg("tile_width") = 32,
               py::arg("tile_height") = 8, py::arg("workers") = 0, py::arg("bloom_threshold") = 12.0f,
               py::arg("use_aces") = true, py::arg("not_use_s_curve") = false,
               "Render and post-process into a C-contiguous uint8 (height, width, 4) NumPy array.");
    module.def("render_rgba8_new", &render_rgba8_new_numpy, py::arg("width"), py::arg("height"),
               py::arg("background"), py::arg("prebaked_disk"), py::arg("color_lut"), py::arg("params"),
               py::arg("tile_width") = 32, py::arg("tile_height") = 8, py::arg("workers") = 0,
               py::arg("bloom_threshold") = 12.0f, py::arg("use_aces") = true,
               py::arg("not_use_s_curve") = false,
               "Allocate and return a post-processed uint8 (height, width, 4) NumPy image.");
}
