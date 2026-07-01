// #include<cooperative_groups.h>
// #include <cuda_pipeline_primitives.h>
// #include "cuda_noise.cuh"  // 不再需要 —— 噪声已预烘焙到 3D 纹理

#define A_SPIN 0.0f

__device__ __forceinline__ float3 normalize(float3 v)
{
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator/(float3 a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float operator*(float3 s, float3 a)
{
    return a.x * s.x + a.y * s.y + a.z * s.z;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

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

    float Teff = (Kelvin - 6500.0f) / (6500.0f * Kelvin * 2.2f);
    float3 RgbColor;
    RgbColor.x = __expf(2.05539304e4f * Teff); 
    RgbColor.y = __expf(2.63463675e4f * Teff); 
    RgbColor.z = __expf(3.30145739e4f * Teff);

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

// =========================================================================
// Kerr-Schild Metric Helper Functions
// =========================================================================

// 计算给定笛卡尔坐标下的 Kerr 椭圆半径 r
__device__ __forceinline__ float kerr_r(float3 x)
{
    float a2 = A_SPIN * A_SPIN;
    float R2 = x.x * x.x + x.y * x.y + x.z * x.z;
    float tmp1 = R2 - a2;
    float tmp2 = sqrtf(tmp1 * tmp1 + 4.0f * a2 * x.z * x.z);
    return sqrtf(0.5f * (tmp1 + tmp2));
}

// 严格的哈密顿方程导数计算 (基于零测地线约束)
__device__ __forceinline__ void kerr_derivs(float3 x, float3 p_cov, float3 &v, float3 &dp, float &r_out)
{
    float a = A_SPIN;
    float a2 = a * a;
    float x2 = x.x * x.x;
    float y2 = x.y * x.y;
    float z2 = x.z * x.z;
    float R2 = x2 + y2 + z2;

    float tmp1 = R2 - a2;
    float tmp2 = sqrtf(tmp1 * tmp1 + 4.0f * a2 * z2);
    float r2 = 0.5f * (tmp1 + tmp2);
    r_out = sqrtf(r2);
    float r4 = r2 * r2;
    float r3 = r2 * r_out;

    float D = r4 + a2 * z2;
    float H = r3 / D;
    
    float A_bl = r2 + a2;
    float lx = (r_out * x.x + a * x.y) / A_bl;
    float ly = (r_out * x.y - a * x.x) / A_bl;
    float lz = x.z / r_out;
    float3 l = make_float3(lx, ly, lz);

    float ldotp = lx * p_cov.x + ly * p_cov.y + lz * p_cov.z;
    float pdotp = p_cov.x * p_cov.x + p_cov.y * p_cov.y + p_cov.z * p_cov.z;

    // 严格求解 p_0 (S = -p_0)
    // (1+H) S^2 - 2H(ldotp) S - (pdotp - H(ldotp)^2) = 0
    // 取物理正根
    float sqrt_term = sqrtf(fmaxf(0.0f, (1.0f + H) * pdotp - H * ldotp * ldotp));
    float S = (H * ldotp + sqrt_term) / (1.0f + H); // S > 0

    // dx^i/dlambda = p^i = p_i + H * l_i * (S - ldotp)
    float term = S - ldotp;
    v = make_float3(p_cov.x + H * lx * term, 
                    p_cov.y + H * ly * term, 
                    p_cov.z + H * lz * term);

    // 严格计算 grad H
    float inv_D2 = 1.0f / (D * D);
    float3 gradH;
    gradH.x = x.x * r_out * (3.0f * a2 * z2 - r4) * inv_D2;
    gradH.y = x.y * r_out * (3.0f * a2 * z2 - r4) * inv_D2;
    gradH.z = x.z * r_out * (3.0f * a2 * z2 - r2 * (r2 + 2.0f * a2)) * inv_D2;

    // dp_i/dlambda = 0.5 * grad H * (S - ldotp)^2
    float dp_scalar = 0.5f * term * term;
    dp = gradH * dp_scalar;
}
// =========================================================================

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

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixel_idx >= imgwidth || pixel_idy >= imgheight)
        return;
    float4 buffer = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float jitterx;
    float jittery;
    float physical_x;
    float physical_y;
    
    const float r_plus = 1.0f + sqrtf(1.0f - A_SPIN * A_SPIN);

    for (int i = 0; i < jitternum; ++i) {

        float2 jit = hammersley(i, jitternum, (unsigned int)pixel_idx, (unsigned int)pixel_idy, 1);
        jitterx = jit.x;
        jittery = jit.y;
        physical_x = (((float)pixel_idx + jitterx) / (float)imgwidth - 0.5f) * physwidth;
        physical_y = (((float)pixel_idy + jittery) / (float)imgheight - 0.5f) * physheight;
        float3 cam_pos = make_float3(cam_pos_x, cam_pos_y, cam_pos_z);

        float3 tmp1 = normalize(make_float3(focal_length,-physical_x,-physical_y));
        #ifndef NO_DEPTH_JITTER
        unsigned int depth_seed = pcg_hash(pixel_idx ^ pcg_hash(pixel_idy ^ pcg_hash(i ^ pcg_hash(frames))));
        float depth_jitter = (float)depth_seed / 4294967296.0f;
        #endif

        // ==================== 严格四标架与光行差计算 ====================
        float r = kerr_r(cam_pos);
        float r2 = r * r;
        float r3 = r2 * r;
        float r4 = r2 * r2;
        float a2 = A_SPIN * A_SPIN;
        float D = r4 + a2 * cam_pos.z * cam_pos.z;
        float H = r3 / D;
        
        // 局部静止观测者的时间基 e_0 = (1/sqrt(1-H), 0,0,0)
        float inv_sqrt_1mH = rsqrtf(1.0f - H);
        
        // Kerr-Schild 零矢量 l_mu
        float A_bl = r2 + a2;
        float3 l = make_float3((r * cam_pos.x + A_SPIN * cam_pos.y) / A_bl, 
                               (r * cam_pos.y - A_SPIN * cam_pos.x) / A_bl, 
                               cam_pos.z / r);
        
        // 构造严格正交于 l 的空间基 (Gram-Schmidt)
        float3 e1 = fwd - l * (l * fwd);
        e1 = normalize(e1);
        float3 e2 = right - l * (l * right) - e1 * (e1 * right);
        e2 = normalize(e2);
        float3 e3 = cross(e1, e2); // 已正交归一
        
        // 观测者在局部静止系下的速度
        float3 beta_vec = make_float3(vfwd, vright, vup);
        float beta_len2 = beta_vec.x * beta_vec.x + beta_vec.y * beta_vec.y + beta_vec.z * beta_vec.z;
        float gamma = rsqrtf(1.0f - beta_len2);
        
        // 将速度映射到全局笛卡尔空间
        float3 beta_global = beta_vec.x * e1 + beta_vec.y * e2 + beta_vec.z * e3;
        
        // 保留与原代码兼容的相机四速度空间分量
        float3 e0_space = gamma * beta_global;
        
        // tmp1 是光子在观测者系下的方向
        float beta_dot_n = beta_vec.x * tmp1.x + beta_vec.y * tmp1.y + beta_vec.z * tmp1.z;
        // 相对静止系的频率 \omega
        float omega = gamma * (1.0f + beta_dot_n);
        
        // 光子在静止系下的方向 (全局笛卡尔表示)
        float3 n_obs = tmp1.x * e1 + tmp1.y * e2 + tmp1.z * e3;
        float3 n_rest;
        if (beta_len2 > 1e-12f) {
            float inv_beta_len2 = 1.0f / beta_len2;
            float3 bnor = beta_global * sqrtf(inv_beta_len2);
            // 严格洛伦兹变换得到静止系下的空间方向
            n_rest = n_obs + (gamma - 1.0f) * (bnor * n_obs) * bnor + gamma * beta_global;
        } else {
            n_rest = n_obs;
        }
        
        // 严格计算全局逆变动量 P^mu = \omega * e_0^\mu + \omega * n_rest
        float P0 = omega * inv_sqrt_1mH;
        float3 P_space = omega * n_rest;
        
        // 严格降维为协变动量 p_i = g_{i0} P^0 + g_{ij} P^j 
        // g_{0i} = -H l_i, g_{ij} = \delta_{ij} + H l_i l_j
        float l_dot_P_space = l.x * P_space.x + l.y * P_space.y + l.z * P_space.z;
        float3 p;
        p.x = -H * l.x * P0 + P_space.x + H * l.x * l_dot_P_space;
        p.y = -H * l.y * P0 + P_space.y + H * l.y * l_dot_P_space;
        p.z = -H * l.z * P0 + P_space.z + H * l.z * l_dot_P_space;
        // ===============================================================
        
        float3 p_init = p;
        float lz = cam_pos.x * p.y - cam_pos.y * p.x;
        
        float factor = 1.0f / sqrtf(1.0f - H); // 红移因子 alpha
        bool flag = true;
        float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int s = 0; s < maxstep && flag; ++s) {
            float3 prev_pos = cam_pos;

            #ifdef USE_RK4    // This is the RK4 method (原代码未改动)
            //step1
            float rmhalf = r - 0.5f;
            float g = -fmaxf(0.0f,upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            float uplsq = upl * upl;
            float uu = 1.0f / (uplsq * uplsq);
            float3 k11 = p * uu;
            float3 k12 = g * cam_pos;

            //calc step length
            bool in_disk_volume = (r > 4.5f && r < 27.0f && fabsf(cam_pos.z) < 3.0f);
            float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
            float current_step = step * fminf(50.0f, fmaxf(0.005f, r - 0.54f)) * zone_multiplier * 5.0f;
            #ifdef PHOTON_RING_OPT
            float dist_to_ps = fabsf(r - 1.866025f);
            float ps_multiplier = 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
            current_step*=ps_multiplier;
            #endif
            //step2
            float stephalf = current_step * 0.5f;
            float3 pos_tmp = cam_pos + (stephalf)*k11;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f,upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k21 = (p + (stephalf)*k12) * uu;
            float3 k22 = pos_tmp * g;

            //step3
            pos_tmp = cam_pos + (stephalf)*k21;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f,upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k31 = (p + (stephalf)*k22) * uu;
            float3 k32 = pos_tmp * g;

            //step4
            pos_tmp = cam_pos + current_step * k31;
            r = length(pos_tmp);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            rmhalf = r - 0.5f;
            g = -fmaxf(0.0f,upl * (2.0f - u) / (rmhalf * rmhalf * rmhalf));
            uplsq = upl * upl;
            uu = 1.0f / (uplsq * uplsq);
            float3 k41 = (p + current_step * k32) * uu;
            float3 k42 = pos_tmp * g;

            //concat results and update var
            stephalf=current_step*0.16666666667f;
            cam_pos = cam_pos + (stephalf) * (k11+k41+2.0f*k21+2.0f*k31);
            p = p + (stephalf) * (k12+k42+2.0f*k22+2.0f*k32);

            //epilogues
            r = length(cam_pos);
            u = 1.0f / (2.0f * r);
            upl = 1.0f + u;
            umi = 1.0f - u;
            n = upl *upl * upl / umi;
            p = normalize(p) * n;
            #else     
            // ==================== RK2 for Kerr-Schild ====================
            // step 1
            float3 k11_v, k11_dp;
            float r_step1;
            kerr_derivs(cam_pos, p, k11_v, k11_dp, r_step1);
            r = r_step1;

            bool in_disk_volume = (r > 4.5f && r < 27.0f && fabsf(cam_pos.z) < 3.0f);
            float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
            float current_step = step * fminf(50.0f, fmaxf(0.005f, r - r_plus)) * zone_multiplier;
            
            #ifdef PHOTON_RING_OPT
            float dist_to_ps = fabsf(r - 1.866025f);
            float ps_multiplier = 0.05f + 0.95f * (dist_to_ps / (dist_to_ps + 0.12f));
            current_step *= ps_multiplier;
            #endif

            // step 2
            float stephalf = current_step * 0.5f;
            float3 pos_tmp = cam_pos + stephalf * k11_v;
            float3 p_tmp = p + stephalf * k11_dp;

            float3 k21_v, k21_dp;
            float r_step2;
            kerr_derivs(pos_tmp, p_tmp, k21_v, k21_dp, r_step2);

            cam_pos = cam_pos + current_step * k21_v;
            p = p + current_step * k21_dp;

            // 更新当前 r
            r = kerr_r(cam_pos);
            // =============================================================
            #endif

            #ifdef RAND_SAMP_DISK
            float rand = rand_float(pcg_hash(pixel_idx ^ pcg_hash(pixel_idy ^ pcg_hash(s ^ pcg_hash(i)))));
            float3 temp = make_float3((cam_pos.x * rand + prev_pos.x * (1.0f-rand)), (cam_pos.y * rand + prev_pos.y * (1.0f-rand)), 0.0f);
            #else
            float3 temp = make_float3((cam_pos.x + prev_pos.x) / 2.0f, (cam_pos.y + prev_pos.y) / 2.0f, 0.0f);
            #endif
            
            float3 mid_pos = make_float3((cam_pos.x + prev_pos.x) * 0.5f, (cam_pos.y + prev_pos.y) * 0.5f, (cam_pos.z + prev_pos.z) * 0.5f);
            float r_disk_mid = kerr_r(mid_pos);
            bool indisk = (r_disk_mid > 4.95f && r_disk_mid < 25.0f && fabsf(mid_pos.z) < 2.5f);

            if (indisk) {
                float r_disk = r_disk_mid;

                float td, pd;
                tdpd(r_disk, &td, &pd);
                float rot = pd * time / td;

                float phi_final = atan2f(temp.y, temp.x) + rot;
                phi_final = fast_mod2pi(phi_final);

                float4 parameters = tex3D<float4>(prebaked_disk,
                                                  phi_final * 0.15915494f,        
                                                  (mid_pos.z / 2.5f) / 2 + 0.5f,  
                                                  (r_disk - 4.9495f) / 20.0505f); 

                float g = fmaxf((fabsf((factor * gamma + p_init * e0_space) / (td - pd * lz)) - 1.0f) * 1.0f + 1.0f, 0.01f);
                float ravg2 = (kerr_r(prev_pos) + r);
                float uuu = 1.0f + 1.0f / (ravg2);
                float g4 = g * g * g * g;
                float step_len = length(cam_pos - prev_pos);

                float k = 2.0f;
                float kzg4 = k * parameters.z * g4;
                float temp_fade = __saturatef((parameters.y * g - 1400.0f) / 500.0f);
                float step_opacity =
                    parameters.x * 1.7f * uuu * uuu * __fmaf_rn(step_len, -__expf(-kzg4 * kzg4), step_len) / g;
                step_opacity *= temp_fade;
                float temp_exp = -__expf(-step_opacity);

                float4 emission = disk_emission_dep(fmaxf(parameters.y * g, 1000.0f), parameters.z * g4, lut_color);

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

            if (r < r_plus + 0.05f || r > 140.0f) {
                flag = false;
            }
        }

        float4 color;
        if (r >= r_plus + 0.05f && !isnan(r)) {

            // 由于积分变量是协变动量 p_i，方向由逆变空间分量 p^i 决定
            // 逆变动量 p^i = p_i + H l^i (S - l.p)，这在 kerr_derivs 计算的 v 中已经求得
            float3 v_dummy, dp_dummy;
            float r_dummy;
            kerr_derivs(cam_pos, p, v_dummy, dp_dummy, r_dummy);
            float3 final_dir = normalize(v_dummy);

            float phi = atan2f(final_dir.y, -final_dir.x);
            float theta = asinf(-final_dir.z);

            float tex_u = phi * 0.1591549f + 0.5f;
            float tex_v = theta * 0.3183099f + 0.5f;

            float4 bkgd = tex2D<float4>(tex_obj, tex_u, tex_v);

            color = accumulated_color + bkgd * (1.0f - accumulated_color.w);

        } else {
            color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
        }

        buffer = buffer + color;
    }
    buffer = buffer * (1.0f / (float)jitternum);
    surf2Dwrite<float4>(make_float4(buffer.x, buffer.y, buffer.z, 0.0),raw_img,pixel_idx*sizeof(float4),pixel_idy);
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