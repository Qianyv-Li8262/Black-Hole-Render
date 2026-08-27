#include "cuda_vec_math_utils.cuh"

__device__ __forceinline__ float3 boost(float3 b, float3 vec, float gamma)
{
    float len2 = b.x * b.x + b.y * b.y + b.z * b.z;
    if (len2 < 1e-12f) {
        return vec;
    }
    float3 bnor = b * rsqrtf(len2);
    float3 vv = vec + (gamma - 1.0f) * (bnor * vec) * bnor;
    return vv;
}

__device__ __forceinline__ float tdot(float r)
{
    return (2.0f * r + 1.0f) / sqrtf(1.0f + 4.0f * r * r - 8.0f * r);
}

__device__ __forceinline__ float phidot(float r)
{
    float a = r * sqrtf(r);
    float b = sqrtf(1.0f + 4.0f * r * r - 8.0f * r);
    return 8.0f * a / b / (2.0f * r + 1.0f) / (2.0f * r + 1.0f);
}
__device__ __forceinline__ float3 KelvinToRgb(float Kelvin)
{
    if (Kelvin < 400.01f)
        return make_float3(0.0f, 0.0f, 0.0f);

    // Teef 转换公式
    float Teff = (Kelvin - 6500.0f) / (6500.0f * Kelvin * 2.2f);
    float3 RgbColor;
    RgbColor.x = __expf(2.05539304e4f * Teff); // Red
    RgbColor.y = __expf(2.63463675e4f * Teff); // Green
    RgbColor.z = __expf(3.30145739e4f * Teff); // Blue

    float max_c = fmaxf(fmaxf(1.5f * RgbColor.x, RgbColor.y), RgbColor.z);
    float BrightnessScale = 1.0f / max_c;

    if (Kelvin < 1000.0f) {
        BrightnessScale *= (Kelvin - 400.0f) / 600.0f;
    }
    return RgbColor * BrightnessScale;
}
__device__ __forceinline__ unsigned int pcg_hash(unsigned int input)
{
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand_float(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)seed / 4294967296.0f;
}

__device__ __forceinline__ float2 hammersley(int i, int jitternum, unsigned int pixel_idx, unsigned int pixel_idy,
                                             unsigned int frames)
{
    float h_x = (float)i / (float)jitternum;
    unsigned int reversed = __brev(i);
    float h_y = (float)reversed * 2.3283064365386963e-10f;

    float shift_x = rand_float(pixel_idx ^ pcg_hash(pixel_idy) ^ (frames * 1919810));
    float shift_y = rand_float(pixel_idy ^ pcg_hash(pixel_idx * 114514) ^ (frames * 1919810));

    float j_x = h_x + shift_x;
    j_x = j_x - floorf(j_x);

    float j_y = h_y + shift_y;
    j_y = j_y - floorf(j_y);

    return make_float2(j_x, j_y);
}

__device__ __forceinline__ float fractf(float x)
{
    return x - floorf(x);
}

__device__ float hash31(float x, float y, float z)
{
    float3 p3 = make_float3(x, y, z);
    p3.x = fractf(p3.x * 0.1031f);
    p3.y = fractf(p3.y * 0.1030f);
    p3.z = fractf(p3.z * 0.0973f);
    float dot_val = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    return fractf((p3.x + p3.y + p3.z) * dot_val);
}

__device__ float3 procedural_stars(float3 dir, int frames)
{
    float3 total_stars = make_float3(0.0f, 0.0f, 0.0f);

    float lod_blend = __expf(-(float)(frames - 1) * 0.5f);

    float sharp1_target = 25.0f;
    float sharp1_moving = 2.0f;
    float s1 = sharp1_moving * lod_blend + sharp1_target * (1.0f - lod_blend);
    float energy_scale1 = s1 / sharp1_target;

    float scale1 = 2000.0f;
    float3 p1 = dir * scale1;
    float3 i1 = make_float3(floorf(p1.x), floorf(p1.y), floorf(p1.z));
    float h1 = hash31(i1.x, i1.y, i1.z);
    float thresh1 = 0.8f;

    if (h1 > thresh1) {
        float offx = hash31(i1.x + 12.f, i1.y + 34.f, i1.z + 56.f);
        float offy = hash31(i1.x + 78.f, i1.y + 90.f, i1.z + 12.f);
        float offz = hash31(i1.x + 34.f, i1.y + 56.f, i1.z + 78.f);
        float dx = (p1.x - i1.x) - offx, dy = (p1.y - i1.y) - offy, dz = (p1.z - i1.z) - offz;
        float dist2 = dx * dx + dy * dy + dz * dz;

        float star_shape = __expf(-dist2 * s1);
        float brightness = ((h1 - thresh1) / (1.0f - thresh1)) * 1.5f * energy_scale1;
        total_stars = total_stars + make_float3(1.0f, 1.0f, 1.0f) * brightness * star_shape;
    }

    float sharp2_target = 18.0f;
    float sharp2_moving = 1.5f;
    float s2 = sharp2_moving * lod_blend + sharp2_target * (1.0f - lod_blend);
    float energy_scale2 = s2 / sharp2_target;

    float scale2 = 1200.0f;
    float3 p2 = dir * scale2;
    float3 i2 = make_float3(floorf(p2.x), floorf(p2.y), floorf(p2.z));
    float h2 = hash31(i2.x + 111.f, i2.y + 222.f, i2.z + 333.f);
    float thresh2 = 0.95f;

    if (h2 > thresh2) {
        float offx = hash31(i2.x + 13.f, i2.y + 35.f, i2.z + 57.f);
        float offy = hash31(i2.x + 79.f, i2.y + 91.f, i2.z + 13.f);
        float offz = hash31(i2.x + 35.f, i2.y + 57.f, i2.z + 79.f);
        float dx = (p2.x - i2.x) - offx, dy = (p2.y - i2.y) - offy, dz = (p2.z - i2.z) - offz;
        float dist2 = dx * dx + dy * dy + dz * dz;

        float star_shape = __expf(-dist2 * s2);
        float brightness = ((h2 - thresh2) / (1.0f - thresh2)) * 8.0f * energy_scale2;

        float r = hash31(i2.x + 1.f, i2.y, i2.z);
        float g = hash31(i2.x, i2.y + 1.f, i2.z);
        float b = hash31(i2.x, i2.y, i2.z + 1.f);
        float3 star_color = normalize(make_float3(r + 0.5f, g + 0.5f, b + 0.8f));

        total_stars = total_stars + star_color * brightness * star_shape;
    }

    return total_stars;
}

__device__ float4 disk_emission_dep(float temp, float intensity, cudaTextureObject_t lut_color)
{

    float4 color = tex2D<float4>(lut_color, (temp - 510.0f) / 20000.0f, 0.5f);

    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}
__device__ float4 disk_emission(float temp, float intensity)
{

    float3 color = KelvinToRgb(temp);

    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}
__device__ __forceinline__ void tdpd(float r, float *td, float *pd)
{
    float r2 = r * r;
    float sqrt_term = sqrtf(1.0f + 4.0f * r2 - 8.0f * r);
    float denom = 2.0f * r + 1.0f;

    *td = denom / sqrt_term;

    float a = r * sqrtf(r);
    *pd = 8.0f * a / (sqrt_term * denom * denom);
}
__device__ __forceinline__ float fast_mod2pi(float val)
{
    return val - floorf(val * 0.159154943f) * 6.283185307f;
}

__device__ __forceinline__ float smoothstepf(float edge0, float edge1, float x)
{
    float t = __saturatef((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}
__device__ __forceinline__ float visible_weight_soft(float lambda_nm)
{

    float low = smoothstepf(330.0f, 410.0f, lambda_nm);
    float high = 1.0f - smoothstepf(740.0f, 850.0f, lambda_nm);
    return low * high;
}

__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t)
{
    return a * (1.0f - t) + b * t;
}

__device__ __forceinline__ float3 wavelength_to_rgb(float lambda_nm)
{

    const float UV_FADE0 = 330.0f;
    const float UV_FADE1 = 410.0f;

    const float IR_FADE0 = 720.0f;
    const float IR_FADE1 = 860.0f;
    float uv_to_visible = smoothstepf(UV_FADE0, UV_FADE1, lambda_nm);
    float visible_to_ir = smoothstepf(IR_FADE0, IR_FADE1, lambda_nm);

    float uv_weight = 1.0f - uv_to_visible;
    float ir_weight = visible_to_ir;
    float visible_weight = uv_to_visible * (1.0f - visible_to_ir);

    float l = fminf(fmaxf(lambda_nm, 380.0f), 780.0f);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;

    if (l < 440.0f) {
        r = -(l - 440.0f) / 60.0f;
        g = 0.0f;
        b = 1.0f;
    } else if (l < 490.0f) {
        r = 0.0f;
        g = (l - 440.0f) / 50.0f;
        b = 1.0f;
    } else if (l < 510.0f) {
        r = 0.0f;
        g = 1.0f;
        b = -(l - 510.0f) / 20.0f;
    } else if (l < 580.0f) {
        r = (l - 510.0f) / 70.0f;
        g = 1.0f;
        b = 0.0f;
    } else if (l < 645.0f) {
        r = 1.0f;
        g = -(l - 645.0f) / 65.0f;
        b = 0.0f;
    } else {
        r = 1.0f;
        g = 0.0f;
        b = 0.0f;
    }

    float edge_factor;
    if (l < 420.0f) {
        edge_factor = 0.3f + 0.7f * (l - 380.0f) / 40.0f;
    } else if (l <= 700.0f) {
        edge_factor = 1.0f;
    } else {
        edge_factor = 0.3f + 0.7f * (780.0f - l) / 80.0f;
    }

    float3 visible_rgb = make_float3(r, g, b) * edge_factor;

    const float3 uv_color = make_float3(0.55f, 0.0f, 1.0f); // 紫色
    const float3 ir_color = make_float3(1.0f, 0.0f, 0.0f);  // 红色

    // 如果你觉得太亮,可以整体降一点:
    const float outside_strength = 0.45f;

    float3 outside_uv = uv_color * outside_strength;
    float3 outside_ir = ir_color * outside_strength;

    float3 out = visible_rgb * visible_weight + outside_uv * uv_weight + outside_ir * ir_weight;

    return out;
}

__device__ __forceinline__ float3 rgb_three_line_frequency_shift(float3 rgb, float g)
{
    g = fmaxf(g, 1e-6f);

    const float lambda_R = 700.0f;
    const float lambda_G = 546.1f;
    const float lambda_B = 435.8f;

    float lR = lambda_R / g;
    float lG = lambda_G / g;
    float lB = lambda_B / g;

    float3 cR = wavelength_to_rgb(lR);
    float3 cG = wavelength_to_rgb(lG);
    float3 cB = wavelength_to_rgb(lB);

    constexpr float green_leaks_to_red = (546.1f - 510.0f) / 70.0f; // ≈ 0.515714
    constexpr float blue_leaks_to_red = (440.0f - 435.8f) / 60.0f;  // ≈ 0.070000

    float3 coeff;
    coeff.y = rgb.y;
    coeff.z = rgb.z;
    coeff.x = rgb.x - green_leaks_to_red * rgb.y - blue_leaks_to_red * rgb.z;

    float3 out = coeff.x * cR + coeff.y * cG + coeff.z * cB;

    float g2 = g * g;
    float g3 = g2 * g;

    out = out * g3;
    out.x = fmaxf(out.x, 0.0f);
    out.y = fmaxf(out.y, 0.0f);
    out.z = fmaxf(out.z, 0.0f);

    return out;
}

extern "C" __global__ void
blackholekernel(cudaSurfaceObject_t raw_img, cudaTextureObject_t tex_obj, cudaTextureObject_t prebaked_disk,
                cudaTextureObject_t lut_color, const float time, const float cam_pos_x, const float cam_pos_y,
                const float cam_pos_z, const float fwd_x, const float fwd_y, const float fwd_z, const float right_x,
                const float right_y, const float right_z, const float up_x, const float up_y, const float up_z,
                const float vfwd, const float vright, const float vup, const int imgwidth, const int imgheight,
                const float physwidth, const float physheight, const float focal_length, const float step,
                const int maxstep, const int jitternum, const int frames

)
{

    float3 fwd = make_float3(fwd_x, fwd_y, fwd_z);
    float3 right = make_float3(right_x, right_y, right_z);
    float3 up = make_float3(up_x, up_y, up_z);

    float3 beta = make_float3(vfwd, vright, vup);
    float gamma = rsqrtf(1 - (beta.x * beta.x + beta.y * beta.y + beta.z * beta.z));
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_idx >= imgwidth || pixel_idy >= imgheight)
        return;
    float4 buffer = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float jitterx;
    float jittery;
    float physical_x;
    float physical_y;
    float3 cam_pos = make_float3(cam_pos_x, cam_pos_y, cam_pos_z);
    float r = length(cam_pos);
    float u = 1.0f / (2.0f * r);
    float upl = 1.0f + u;
    float umi = 1.0f - u;
    float factor = upl / umi;
    float n = upl * upl * upl / umi;
    float3 beta_global = beta.x * fwd + beta.y * right + beta.z * up;
    float3 e0 = beta_global * gamma / (upl * upl);
    float3 e1_fwd = boost(beta, make_float3(1.0f, 0.0f, 0.0f), gamma);
    e1_fwd = (e1_fwd.x * fwd + e1_fwd.y * right + e1_fwd.z * up) / (upl * upl);
    float3 e2_right = boost(beta, make_float3(0.0f, 1.0f, 0.0f), gamma);
    e2_right = (e2_right.x * fwd + e2_right.y * right + e2_right.z * up) / (upl * upl);
    float3 e3_up = boost(beta, make_float3(0.0f, 0.0f, 1.0f), gamma);
    e3_up = (e3_up.x * fwd + e3_up.y * right + e3_up.z * up) / (upl * upl);
    for (int i = 0; i < jitternum; ++i) {

        float2 jit = hammersley(i, jitternum, (unsigned int)pixel_idx, (unsigned int)pixel_idy, 1);
        jitterx = jit.x;
        jittery = jit.y;
        physical_x = (((float)pixel_idx + jitterx) / (float)imgwidth - 0.5f) * physwidth;
        physical_y = (((float)pixel_idy + jittery) / (float)imgheight - 0.5f) * physheight;
        float3 cam_pos = make_float3(cam_pos_x, cam_pos_y, cam_pos_z);
        float delta_t = 0.0f;

        // float r_pixel = sqrtf(physical_x * physical_x + physical_y * physical_y);
        // float theta = r_pixel / focal_length;
        // float sin_theta, cos_theta;
        // sincosf(theta, &sin_theta, &cos_theta);
        // float c_phi = 0.0f;
        // float s_phi = 0.0f;
        // if (r_pixel > 1e-6f) {
        //     c_phi = physical_x / r_pixel;
        //     s_phi = physical_y / r_pixel;
        // }
        // float3 tmp1 = make_float3(cos_theta, (sin_theta * c_phi), -(sin_theta * s_phi));
        float3 tmp1 = normalize(make_float3(focal_length, physical_x, -physical_y));
#ifndef NO_DEPTH_JITTER
        unsigned int depth_seed = pcg_hash(pixel_idx ^ pcg_hash(pixel_idy ^ pcg_hash(i ^ pcg_hash(frames))));
        float depth_jitter = (float)depth_seed / 4294967296.0f;
#endif

        float r = length(cam_pos);
        float u = 1.0f / (2.0f * r);
        float upl = 1.0f + u;
        float umi = 1.0f - u;
        float factor = upl / umi;
        float n = upl * upl * upl / umi;
        float3 d = normalize(tmp1.x * e1_fwd + tmp1.y * e2_right + tmp1.z * e3_up - 1.0f * e0);
#ifndef NO_DEPTH_JITTER
        cam_pos = cam_pos + d * (depth_jitter * fmaxf((r - 1.5f), 0.0f) / 10.0f * step);
#endif
        float3 p = d * n;
        float3 p_init = p;
        float lz = cam_pos.x * p.y - cam_pos.y * p.x;
        bool flag = true;
        float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int s = 0; s < maxstep && flag; ++s) {
            float3 prev_pos = cam_pos;
            float prev_dt = delta_t;

#ifdef USE_RK4 // This is the RK4 method
            // step1
            float rmhalf = r - 0.5f;
            float g = -fmaxf(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            float uplsq = upl * upl;
            float uu = 1.0f / (uplsq * uplsq);
            float3 k11 = p * uu;
            float3 k12 = g * cam_pos;
            float k_t1 = uplsq / (umi * umi);

            // calc step length
            bool in_disk_volume = (r > 4.5f && r < 27.0f && fabsf(cam_pos.z) < 3.0f);
            float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
            float current_step =
                step * fminf(50.0f, fmaxf(0.005f, r - 0.54f)) * zone_multiplier * 5.0f; // 这里RK4放宽步长为5倍
#ifdef PHOTON_RING_OPT
            float dist_to_ps = fabsf(r - 1.866025f);
            float ps_multiplier = 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
            current_step *= ps_multiplier;
#endif
            // step2
            float stephalf = current_step * 0.5f;
            float3 pos_tmp = cam_pos + (stephalf)*k11;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k21 = (p + (stephalf)*k12) * uu;
            float3 k22 = pos_tmp * g;
            float k_t2 = uplsq / (umi * umi);

            // step3
            pos_tmp = cam_pos + (stephalf)*k21;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k31 = (p + (stephalf)*k22) * uu;
            float3 k32 = pos_tmp * g;
            float k_t3 = uplsq / (umi * umi);

            // step4
            pos_tmp = cam_pos + current_step * k31;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f, upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k41 = (p + current_step * k32) * uu;
            float3 k42 = pos_tmp * g;
            float k_t4 = uplsq / (umi * umi);

            // concat results and update var
            stephalf = current_step * 0.16666666667f;
            cam_pos = cam_pos + (stephalf) * (k11 + k41 + 2.0f * k21 + 2.0f * k31);
            p = p + (stephalf) * (k12 + k42 + 2.0f * k22 + 2.0f * k32);
            delta_t += stephalf * (k_t1 + k_t4 + 2.0f * k_t2 + 2.0f * k_t3);

            // epilogues
            r = length(cam_pos);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            n = upl * upl * upl / umi;
            p = normalize(p) * n;
#else // This is the RK2 method
      // step 1
            float rmhalf = r - 0.5f;
            float g = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
            float uplsq = upl * upl;
            float uu = 1.0f / (uplsq * uplsq);
            float3 k11 = p * uu;
            float3 k12 = g * cam_pos;
            // float k_t1 = uplsq/(umi*umi);

            bool in_disk_volume = (r > 4.5f && r < 37.0f && fabsf(cam_pos.z) < 3.0f);
            float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
            float current_step = step * fminf(50.0f, fmaxf(0.005f, r - 0.54f)) * zone_multiplier;
#ifdef PHOTON_RING_OPT
            float dist_to_ps = fabsf(r - 1.866025f);
            float ps_multiplier = 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
            current_step *= ps_multiplier;
#endif

            // step 2
            float stephalf = current_step * 0.5f;
            float3 pos_tmp = cam_pos + (stephalf)*k11;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf);
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k21 = (p + (stephalf)*k12) * uu;
            float3 k22 = pos_tmp * g;
            float k_t2 = uplsq / (umi * umi);

            cam_pos = cam_pos + (current_step) * (k21);
            p = p + (current_step) * (k22);
            delta_t += current_step * k_t2;
            r = length(cam_pos);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            n = upl * upl * upl / umi;
            p = normalize(p) * n;
#endif
#ifdef RAND_SAMP_DISK
            float rand = rand_float(pcg_hash(pixel_idx ^ pcg_hash(pixel_idy ^ pcg_hash(s ^ pcg_hash(i)))));
            // float3 temp = make_float3((cam_pos.x * rand + prev_pos.x * (1.0f - rand)),
            //                           (cam_pos.y * rand + prev_pos.y * (1.0f - rand)), 0.0f);
            float3 temp = cam_pos * rand + prev_pos * (1.0f - rand);
            float t_use = (delta_t * rand + prev_dt * (1.0f - rand));
#else
            float3 temp = (cam_pos + prev_pos) * 0.5f;
            float t_use = (prev_dt + delta_t) * 0.5f;
#endif
            float r_disk_sq = temp.x * temp.x + temp.y * temp.y;
            bool indisk = (r_disk_sq > 24.4974f && r_disk_sq < 1225.0f && fabsf(cam_pos.z) < 2.5f);

            if (indisk) {
                float r_disk = sqrtf(r_disk_sq); // 差动旋转

                float td, pd;
                tdpd(r_disk, &td, &pd);
                float td2, pd2;
                tdpd(7.0f, &td2, &pd2);
                // float rot = pd * 200.0f / td + pd2 * (time - t_use) / td2;
                float rot = pd * (time - t_use) / td;

                float phi_final = atan2f(temp.y, temp.x) + rot;
                phi_final = fast_mod2pi(phi_final);

                float4 parameters = tex3D<float4>(prebaked_disk,
                                                  phi_final * 0.15915494f,        // x → phi
                                                  (temp.z / 2.5f) / 2 + 0.5f,     // y → z
                                                  (r_disk - 4.9495f) / 30.0505f); // z → r_disk

                float g = fmaxf((fabsf((factor * gamma + p_init * e0) / (td - pd * lz)) - 1.0f) * 1.0f + 1.0f, 0.01f);
                float ravg2 = (length(prev_pos) + r);
                float uuu = 1.0f + 1.0f / (ravg2);
                float g4 = g * g * g * g;
                float step_len = length(cam_pos - prev_pos);

                float k = 2.0f;
                float kzg4 = k * parameters.z;
                // float intensity_factor = 1.0f - __expf(-kzg4
                // * kzg4);
#ifdef OPACITY_CHANGE
                // 冷区凝聚增强: T_eff 低 → opacity 乘 (1 + COLD_BOOST);T_eff 高 → 无加成
                // T_COLD=2000K 以下满档,T_WARM=5000K 以上无加成,中间线性
                // OPACITY_SCALE 把全局不透明度从 1.7 提到 3.0,整体更厚实
                // COLD_BOOST=2.0: 冷区最多 3× 不透明度,塑造冷云团凝聚感
                float T_eff = parameters.y * g;
                float cold_factor = 1.0f + 2.0f * __saturatef((5000.0f - T_eff) / 3000.0f);
                float step_opacity = parameters.x * 3.0f * uuu * uuu *
                                     __fmaf_rn(step_len, -__expf(-kzg4 * kzg4), step_len) / g * cold_factor;
#else
                // float temp_fade = __saturatef((parameters.y * g - 1400.0f) / 500.0f);
                float temp_fade = 1.0f;
                // float step_opacity = parameters.x * 1.7f *
                // uuu * uuu * step_len * intensity_factor / g;
                float step_opacity =
                    parameters.x * 1.7f * uuu * uuu * __fmaf_rn(step_len, -__expf(-kzg4 * kzg4), step_len) / g;
                step_opacity *= temp_fade;
#endif
                // float alpha = 1.0f - __expf(-step_opacity);
                float temp_exp = -__expf(-step_opacity);
                // float transmittance = 1.0f -
                // accumulated_color.w;
#ifndef ACCRE_DISK_DOPPLER_FOLLOW_BKGD // DO NOT USE UNLESS U KNOW WHAT UR DOING
                float4 emission = disk_emission_dep(fmaxf(parameters.y * g, 1000.0f), parameters.z * g4, lut_color);
#else
                float4 color = tex2D<float4>(lut_color, (parameters.y - 510.0f) / 20000.0f, 0.5f);
                float intensity = parameters.z * g4;
                float3 rgb = make_float3(color.x, color.y, color.z);
                float3 shifted_rgb = rgb_three_line_frequency_shift(rgb, g);
                color.x = shifted_rgb.x;
                color.y = shifted_rgb.y;
                color.z = shifted_rgb.z;
                float4 emission = make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
#endif

                float temp_calc = __fmaf_rn(emission.x, -accumulated_color.w, emission.x);
                accumulated_color.x += __fmaf_rn(temp_calc, temp_exp, temp_calc);
                temp_calc = __fmaf_rn(emission.y, -accumulated_color.w, emission.y);
                accumulated_color.y += __fmaf_rn(temp_calc, temp_exp, temp_calc);
                temp_calc = __fmaf_rn(emission.z, -accumulated_color.w, emission.z);
                accumulated_color.z += __fmaf_rn(temp_calc, temp_exp, temp_calc);
                temp_calc = 1 - accumulated_color.w;
                accumulated_color.w += __fmaf_rn(temp_calc, temp_exp, temp_calc);
                if (accumulated_color.w > 0.99f) {
                    flag = false;
                }
            }

            if (r < 0.55f || r > 140.0f) {
                flag = false;
            }
            // if (pixel_idx == 2000 && pixel_idy == 1500){
            //     printf("Debug,xyz: %f,%f,%f,step: %f,indisk:%d,steps:
            //     %d,buffer1:%f\n",cam_pos.x,cam_pos.y,cam_pos.z,current_step,indisk,s,accumulated_color.x);
            // }
        }

        float4 color;
        if (r >= 0.55f && !isnan(r)) {

            float3 final_dir = normalize(p);

            float phi = atan2f(final_dir.y, -final_dir.x);
            float theta = asinf(-final_dir.z);

            float tex_u = phi * 0.1591549f + 0.5f;
            float tex_v = theta * 0.3183099f + 0.5f;

            float4 bkgd = tex2D<float4>(tex_obj, tex_u, tex_v);
#ifndef NO_BKGD_DOPPLER
            // 静止无穷远星空的频移因子。
            // 你的 p_init 是反向追踪动量,所以这里和吸积盘 numerator 保持一致。
            float g_sky = factor * gamma + p_init * e0;
            g_sky = fabsf(g_sky);
            // 避免极端数值。
            // 波长本身已经 clamp 到 [380,780],
            // 但 g^3 亮度仍可能爆,所以这里可以给一个上限。
            // 如果你希望完全物理夸张,可以删掉 fminf。
            g_sky = fmaxf(g_sky, 1e-4f);
            g_sky = fminf(g_sky, 20.0f);
            // 如果 tex_obj 已经是 linear RGB,直接使用。
            // 如果 tex_obj 是普通 sRGB 贴图,理论上应该先 sRGB -> linear。
            float3 bkgd_rgb = make_float3(bkgd.x, bkgd.y, bkgd.z);
            // 三谱线艺术化频移
            float3 shifted_rgb = rgb_three_line_frequency_shift(bkgd_rgb, g_sky);
            bkgd.x = shifted_rgb.x;
            bkgd.y = shifted_rgb.y;
            bkgd.z = shifted_rgb.z;
            // float g3 = g_sky * g_sky * g_sky;
            // bkgd.x = bkgd_rgb.x * g3;
            // bkgd.y = bkgd_rgb.y * g3;
            // bkgd.z = bkgd_rgb.z * g3;
#endif

            color = accumulated_color + bkgd * (1.0f - accumulated_color.w);

        } else {

            color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
        }

        buffer = buffer + color;
    }
    buffer = buffer * (1.0f / (float)jitternum);
    // int pixel_index = (pixel_idy * imgwidth + pixel_idx);
    surf2Dwrite<float4>(make_float4(buffer.x, buffer.y, buffer.z, 0.0), raw_img, pixel_idx * sizeof(float4), pixel_idy);
}

extern "C" __global__ void taaColorClampingKernel(const float4 *currentFrame, const float4 *prevFrame,
                                                  float4 *outputFrame, int width, int height, float alpha, int frames)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float3 min_color = make_float3(1e10f, 1e10f, 1e10f);
    float3 max_color = make_float3(-1e10f, -1e10f, -1e10f);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = fmaxf(0, fminf(x + dx, width - 1));
            int ny = fmaxf(0, fminf(y + dy, height - 1));

            float4 c = currentFrame[ny * width + nx];

            min_color.x = fminf(min_color.x, c.x);
            min_color.y = fminf(min_color.y, c.y);
            min_color.z = fminf(min_color.z, c.z);

            max_color.x = fmaxf(max_color.x, c.x);
            max_color.y = fmaxf(max_color.y, c.y);
            max_color.z = fmaxf(max_color.z, c.z);
        }
    }

    float4 prev_col = prevFrame[y * width + x];

    float3 clamped_prev;
    clamped_prev.x = fmaxf(min_color.x, fminf(prev_col.x, max_color.x));
    clamped_prev.y = fmaxf(min_color.y, fminf(prev_col.y, max_color.y));
    clamped_prev.z = fmaxf(min_color.z, fminf(prev_col.z, max_color.z));

    float4 curr_col = currentFrame[y * width + x];
    float4 final_col;
    final_col.x = curr_col.x * alpha + clamped_prev.x * (1.0f - alpha);
    final_col.y = curr_col.y * alpha + clamped_prev.y * (1.0f - alpha);
    final_col.z = curr_col.z * alpha + clamped_prev.z * (1.0f - alpha);
    final_col.w = curr_col.w;

    outputFrame[y * width + x] = final_col * frames;
}
