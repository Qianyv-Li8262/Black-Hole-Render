#pragma once
#define PI 3.1415926535897932384F
#include "cuda_noise.cuh"

namespace accretionCloud
{

__device__ __forceinline__ unsigned int mixSeed(unsigned int seed, unsigned int salt)
{
    return cudaNoise::hash(seed ^ salt);
}

__device__ __forceinline__ float3 scale3(float3 p, float s)
{
    return make_float3(p.x * s, p.y * s, p.z * s);
}

__device__ __forceinline__ float3 rotateNoiseDomain(float3 p)
{
    // Approximately orthonormal rotation used between octaves to suppress
    // grid alignment without changing the characteristic feature size.
    return make_float3(0.000000f * p.x + 0.800000f * p.y + 0.600000f * p.z,
                       -0.800000f * p.x + 0.360000f * p.y - 0.480000f * p.z,
                       -0.600000f * p.x - 0.480000f * p.y + 0.640000f * p.z);
}

__device__ __forceinline__ float cloudFBM(float3 p, unsigned int seed, int octaves, float lacunarity, float persistence)
{
    float sum = 0.0f;
    float amplitude = 1.0f;
    float weightSum = 0.0f;

    for (int i = 0; i < octaves; ++i) {
        const int octaveSeed = static_cast<int>(mixSeed(seed, 0x9E3779B9u * static_cast<unsigned int>(i + 1)));

        sum += cudaNoise::simplexNoise(p, 1.0f, octaveSeed) * amplitude;
        weightSum += amplitude;

        p = scale3(rotateNoiseDomain(p), lacunarity);
        amplitude *= persistence;
    }

    return sum / fmaxf(weightSum, 1.0e-6f);
}

__device__ __forceinline__ float cloudRidgedFBM(float3 p, unsigned int seed, int octaves, float lacunarity,
                                                float persistence)
{
    float sum = 0.0f;
    float amplitude = 1.0f;
    float weightSum = 0.0f;
    float previous = 1.0f;

    for (int i = 0; i < octaves; ++i) {
        const int octaveSeed = static_cast<int>(mixSeed(seed, 0x85EBCA6Bu * static_cast<unsigned int>(i + 1)));

        float n = cudaNoise::simplexNoise(p, 1.0f, octaveSeed);
        float ridge = 1.0f - fabsf(n);
        ridge = ridge * ridge;

        // Successive ridges reinforce existing large-scale wisps instead of
        // filling the volume with uniform high-frequency noise.
        ridge *= 0.45f + 0.55f * previous;
        previous = ridge;

        sum += ridge * amplitude;
        weightSum += amplitude;

        p = scale3(rotateNoiseDomain(p), lacunarity);
        amplitude *= persistence;
    }

    return __saturatef(sum / fmaxf(weightSum, 1.0e-6f));
}

// Returns cloud density in [0, 1].
//
// position uses a right-handed Cartesian space. The implied accretion plane is
// XY and Z is the vertical axis. No radial, inner-hole, outer-edge, or vertical
// density envelope is applied here.
__device__ __forceinline__ float accretionDiskCloudDensity(int seed, float3 position)
{
    const unsigned int baseSeed = static_cast<unsigned int>(seed);

    // Global feature scale.
    float3 p = scale3(position, 0.23f);

    // Seamless radius-dependent rotation. This bends broad structures into
    // curved, differential-rotation-like streaks without using atan2().
    const float radius = sqrtf(p.x * p.x + p.y * p.y);

    const float lowFrequencyBend =
        cudaNoise::simplexNoise(scale3(p, 0.18f), 1.0f, static_cast<int>(mixSeed(baseSeed, 0xA511E9B3u)));

    const float angle = 1.65f * log1pf(radius) + 0.30f * lowFrequencyBend;

    float sinAngle;
    float cosAngle;
    sincosf(angle, &sinAngle, &cosAngle);

    float3 q = make_float3(cosAngle * p.x - sinAngle * p.y, sinAngle * p.x + cosAngle * p.y, p.z);

    // Low-frequency vector domain warp creates large rolling cloud masses.
    const float3 warpPosition = scale3(q, 0.42f);

    const float warpX = cudaNoise::simplexNoise(warpPosition, 1.0f, static_cast<int>(mixSeed(baseSeed, 0x243F6A88u)));

    const float warpY = cudaNoise::simplexNoise(warpPosition, 1.0f, static_cast<int>(mixSeed(baseSeed, 0xB7E15162u)));

    const float warpZ = cudaNoise::simplexNoise(warpPosition, 1.0f, static_cast<int>(mixSeed(baseSeed, 0xC6EF3720u)));

    q.x += warpX * 0.72f;
    q.y += warpY * 0.72f;
    q.z += warpZ * 0.38f;

    // Anisotropic sampling stretches structures along the warped flow
    // direction and keeps vertical detail comparatively compressed.
    const float3 flowDomain = make_float3(q.x * 0.46f, q.y * 1.12f, q.z * 1.38f);

    const float macroSigned = cloudFBM(flowDomain, mixSeed(baseSeed, 0x3C6EF372u), 5, 2.03f, 0.52f);

    const float macro = __saturatef(0.5f + 0.5f * macroSigned);

    const float wisps = cloudRidgedFBM(make_float3(flowDomain.x * 1.30f, flowDomain.y * 1.08f, flowDomain.z * 1.30f),
                                       mixSeed(baseSeed, 0xDAA66D2Bu), 4, 2.17f, 0.48f);

    // A differently oriented secondary layer prevents the cloud field from
    // degenerating into perfectly parallel streaks.
    const float secondarySigned =
        cloudFBM(make_float3(0.58f * q.x + 0.34f * q.y, -0.34f * q.x + 0.58f * q.y, 0.96f * q.z),
                 mixSeed(baseSeed, 0x78DDE6E4u), 4, 2.08f, 0.50f);

    const float secondary = __saturatef(0.5f + 0.5f * secondarySigned);

    float cloudField = macro * 0.69f + wisps * 0.21f + secondary * 0.10f;

    // Coverage shaping: preserves broad fog banks while opening irregular
    // transparent gaps. This is local density shaping, not a spatial fade.
    float density = __saturatef((cloudField - 0.38f) / 0.29f);
    density = density * density * (3.0f - 2.0f * density);

    // Fine breakup only modulates already-existing cloud masses.
    const float fineSigned = cloudFBM(scale3(flowDomain, 2.75f), mixSeed(baseSeed, 0x1715609Du), 3, 2.21f, 0.43f);

    const float fine = __saturatef(0.5f + 0.5f * fineSigned);
    density *= __saturatef(0.57f + 0.78f * fine);

    // Slightly sharpen dense cores while retaining soft cloud boundaries.
    density = density * density * (2.15f - 1.15f * density);

    return __saturatef(density);
}

__device__ __forceinline__ float cloudSmoothstep(float x, float edge0, float edge1)
{
    const float t = __saturatef((x - edge0) / fmaxf(edge1 - edge0, 1.0e-6f));
    return t * t * (3.0f - 2.0f * t);
}

// 对两个密度场进行类似体积并集的组合。
// 与直接相加相比不容易过曝，也能保留云块互相叠加的感觉。
__device__ __forceinline__ float cloudUnion(float a, float b)
{
    return 1.0f - (1.0f - a) * (1.0f - b);
}

// Returns cloud density in [0, 1].
//
// position uses a right-handed Cartesian space.
// XY is the accretion plane and Z is the vertical axis.
// Radial and vertical envelopes are still applied outside this function.
__device__ __forceinline__ float accretionDiskCloudDensity2(int seed, float3 position)
{
    const unsigned int baseSeed = static_cast<unsigned int>(seed);

    // 数值越小，云块越大。
    // 原来是 0.23，现在降低频率以获得更大尺度的云团。
    float3 p = scale3(position, 0.145f);

    const float radius = sqrtf(p.x * p.x + p.y * p.y);

    // 只保留很弱的盘面旋转，避免生成明显的螺旋丝带。
    const float bendNoise =
        cudaNoise::simplexNoise(scale3(p, 0.20f), 1.0f, static_cast<int>(mixSeed(baseSeed, 0xA511E9B3u)));

    const float angle = 0.28f * log1pf(radius) + 0.10f * bendNoise;

    float sinAngle;
    float cosAngle;
    sincosf(angle, &sinAngle, &cosAngle);

    float3 q = make_float3(cosAngle * p.x - sinAngle * p.y, sinAngle * p.x + cosAngle * p.y, p.z);

    // 大尺度、近似各向同性的域扭曲。
    // 域扭曲负责让云块边界翻滚，但不再把它们拉成长条。
    const float3 warpDomain = scale3(q, 0.34f);

    const float warpX = cudaNoise::simplexNoise(warpDomain, 1.0f, static_cast<int>(mixSeed(baseSeed, 0x243F6A88u)));

    const float warpY = cudaNoise::simplexNoise(warpDomain, 1.0f, static_cast<int>(mixSeed(baseSeed, 0xB7E15162u)));

    const float warpZ = cudaNoise::simplexNoise(warpDomain, 1.0f, static_cast<int>(mixSeed(baseSeed, 0xC6EF3720u)));

    q.x += warpX * 0.78f;
    q.y += warpY * 0.78f;
    q.z += warpZ * 0.55f;

    // ------------------------------------------------------------
    // 1. 超低频集中区域
    // ------------------------------------------------------------
    // 这一层决定大块云团主要出现在哪里，使云不是均匀铺满整个盘面。
    const float clusterSigned =
        cloudFBM(make_float3(q.x * 0.32f, q.y * 0.32f, q.z * 0.28f), mixSeed(baseSeed, 0x6A09E667u), 3, 1.91f, 0.56f);

    const float clusterNoise = __saturatef(0.5f + 0.5f * clusterSigned);

    // 阈值越高，云越集中、空白区域越多。
    const float clusterMask = cloudSmoothstep(clusterNoise, 0.36f, 0.63f);

    // ------------------------------------------------------------
    // 2. 第一层大云块
    // ------------------------------------------------------------
    const float layerASigned =
        cloudFBM(make_float3(q.x * 0.72f, q.y * 0.72f, q.z * 0.66f), mixSeed(baseSeed, 0x3C6EF372u), 4, 1.96f, 0.53f);

    const float layerANoise = __saturatef(0.5f + 0.5f * layerASigned);

    const float layerA = cloudSmoothstep(layerANoise, 0.43f, 0.67f);

    // ------------------------------------------------------------
    // 3. 第二层大云块
    // ------------------------------------------------------------
    // 使用不同方向和偏移，避免两层云完全重合。
    const float3 layerBDomain =
        make_float3((-0.36f * q.x + 0.93f * q.y + 7.13f) * 0.86f, (-0.93f * q.x - 0.36f * q.y - 3.71f) * 0.86f,
                    (0.92f * q.z + 2.93f) * 0.86f);

    const float layerBSigned = cloudFBM(layerBDomain, mixSeed(baseSeed, 0xBB67AE85u), 4, 2.01f, 0.51f);

    const float layerBNoise = __saturatef(0.5f + 0.5f * layerBSigned);

    const float layerB = cloudSmoothstep(layerBNoise, 0.45f, 0.69f);

    // ------------------------------------------------------------
    // 4. 中等尺度的附属云块
    // ------------------------------------------------------------
    // 这一层产生附着在大云团周围的小型云瓣，而不是细丝。
    const float3 layerCDomain =
        make_float3((0.78f * q.x + 0.62f * q.y - 5.27f) * 1.08f, (-0.62f * q.x + 0.78f * q.y + 8.31f) * 1.08f,
                    (0.88f * q.z - 4.17f) * 1.08f);

    const float layerCSigned = cloudFBM(layerCDomain, mixSeed(baseSeed, 0xA54FF53Au), 3, 2.05f, 0.49f);

    const float layerCNoise = __saturatef(0.5f + 0.5f * layerCSigned);

    const float layerC = cloudSmoothstep(layerCNoise, 0.48f, 0.70f);

    // ------------------------------------------------------------
    // 5. 将多层云块按体积并集组合
    // ------------------------------------------------------------
    float cloudMass = layerA;
    cloudMass = cloudUnion(cloudMass, layerB * 0.82f);
    cloudMass = cloudUnion(cloudMass, layerC * 0.56f);

    // 两个大云层重叠的位置增加密度，形成“一块块叠起来”的核心。
    const float overlapCore = layerA * layerB;

    // 集中掩膜之外只留下少量薄雾，主体集中在低频区域内。
    float density = cloudMass * (0.08f + 0.92f * clusterMask);
    density += 0.18f * overlapCore * clusterMask;
    density = __saturatef(density);

    // 将低密度噪声清掉，同时保持边界柔软。
    density = cloudSmoothstep(density, 0.15f, 0.76f);

    // ------------------------------------------------------------
    // 6. 高频细节只用于边缘破碎
    // ------------------------------------------------------------
    // 普通 FBM，不使用 ridged FBM，避免重新产生脊线和丝带。
    const float fineSigned =
        cloudFBM(make_float3(q.x * 2.10f, q.y * 2.10f, q.z * 1.85f), mixSeed(baseSeed, 0x510E527Fu), 3, 2.13f, 0.43f);

    const float fine = __saturatef(0.5f + 0.5f * fineSigned);

    // 调制幅度不要太大，否则大云块会再次被切碎。
    density *= 0.72f + 0.38f * fine;
    density = __saturatef(density);

    // 轻微增强核心，但比原算法更温和，保留雾状过渡。
    density *= 0.74f + 0.26f * density;

    return __saturatef(density);
}

} // namespace accretionCloud

__device__ __forceinline__ float smoothstep1(float x, float edge0, float edge1)
{
    float xx = __saturatef((x - edge0) / (edge1 - edge0));
    return xx * xx * (3.0f - 2.0f * xx);
}

__device__ __forceinline__ float smoothstep2(float x, float edge0, float edge1)
{
    return 1.0f - smoothstep1(x, edge0, edge1);
}

__device__ __forceinline__ float lerpf(float edge0, float edge1, float param)
{
    return param * (edge1 - edge0) + edge0;
}

extern "C" __global__ void get_cloud_density(float *__restrict__ output, float r_min, float r_max, float z_box_max,
                                             int phi_samples, int z_samples, int r_samples, int seed

)
{ // phi, z, r
    int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int r_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (r_idx >= r_samples || z_idx >= z_samples || phi_idx >= phi_samples)
        return;
    float r = lerpf(r_min, r_max, static_cast<float>(r_idx) / static_cast<float>(r_samples));
    float z = lerpf(-z_box_max, z_box_max, static_cast<float>(z_idx) / static_cast<float>(z_samples));
    float phi = lerpf(0, 2 * PI, static_cast<float>(phi_idx) / static_cast<float>(phi_samples));
    float cos;
    float sin;
    sincosf(phi, &sin, &cos);
    float3 cart_pos = make_float3(r * cos, r * sin, z); // another calc way ?
    float density_raw = accretionCloud::accretionDiskCloudDensity2(seed, cart_pos);
    float z_variation_to_r = lerpf(0.5f, 3.5f, static_cast<float>(r_idx) / static_cast<float>(r_samples));
    float z_envelope = smoothstep2(z, z_variation_to_r, z_variation_to_r + 1.0f) *
                       smoothstep1(z, -1.0f - z_variation_to_r, -z_variation_to_r);
    // float z_envelope = __expf(-z*z*3/z_variation_to_r);
    float r_envelope = smoothstep1(r, r_min, r_min + 0.2f) * smoothstep2(r, r_max - 10.0f, r_max);
    float density = density_raw * z_envelope * r_envelope;
    density *= 0.7f; // knob for scaling
    float T0 = 18400.0f;
    float x = r - 4.9495f;
    float base_temp = T0 * powf(x / 0.8f, -1.0f);
    base_temp = fmaxf(base_temp * ((density_raw - 0.5f) * 0.1f + 1.0f), 0.0f); // temp_fade considering.
    float intensity = 60.0f * powf(4 / (r - 4.8f), 2.0f);
    intensity *= ((density_raw - 0.5f) * 0.1f + 1.0f);
    const int base = ((phi_idx * z_samples + z_idx) * r_samples + r_idx) * 4u;
    output[base + 0] = base_temp; // temperature
    output[base + 1] = density;   // density
    output[base + 2] = intensity; // intensity
}

extern "C" __global__ void compute_density_gradient(float *__restrict__ output, float r_min, float r_max,
                                                    float z_box_max, int phi_samples, int z_samples, int r_samples)
{
    const int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int r_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (phi_idx >= phi_samples || z_idx >= z_samples || r_idx >= r_samples) {
        return;
    }

    const float dphi = 2.0f * PI / static_cast<float>(phi_samples);
    const float dz = 2.0f * z_box_max / static_cast<float>(z_samples);
    const float dr = (r_max - r_min) / static_cast<float>(r_samples);

    const float r = lerpf(r_min, r_max, static_cast<float>(r_idx) / static_cast<float>(r_samples));

    const int center_base = ((phi_idx * z_samples + z_idx) * r_samples + r_idx) * 4;

    const int phi_prev = (phi_idx == 0) ? (phi_samples - 1) : (phi_idx - 1);
    const int phi_next = (phi_idx + 1 == phi_samples) ? 0 : (phi_idx + 1);

    const int phi_prev_base = ((phi_prev * z_samples + z_idx) * r_samples + r_idx) * 4;
    const int phi_next_base = ((phi_next * z_samples + z_idx) * r_samples + r_idx) * 4;
    const float density_phi_prev = output[phi_prev_base + 1];
    const float density_phi_next = output[phi_next_base + 1];
    const float safe_r = fmaxf(fabsf(r), 1.0e-6f);

    float grad_phi = 0.0f;
    if (phi_samples > 1) {
        grad_phi = (density_phi_next - density_phi_prev) / (2.0f * dphi * safe_r);
    }

    float grad_z = 0.0f;

    if (z_samples > 1) {
        if (z_idx == 0) {
            // Forward difference at lower z boundary.
            const int z_next_base = ((phi_idx * z_samples + (z_idx + 1)) * r_samples + r_idx) * 4;
            const float density_center = output[center_base + 1];
            const float density_z_next = output[z_next_base + 1];
            grad_z = (density_z_next - density_center) / dz;
        } else if (z_idx == z_samples - 1) {
            // Backward difference at upper z boundary.
            const int z_prev_base = ((phi_idx * z_samples + (z_idx - 1)) * r_samples + r_idx) * 4;
            const float density_center = output[center_base + 1];
            const float density_z_prev = output[z_prev_base + 1];
            grad_z = (density_center - density_z_prev) / dz;
        } else {
            // Central difference in the interior.
            const int z_prev_base = ((phi_idx * z_samples + (z_idx - 1)) * r_samples + r_idx) * 4;
            const int z_next_base = ((phi_idx * z_samples + (z_idx + 1)) * r_samples + r_idx) * 4;
            const float density_z_prev = output[z_prev_base + 1];
            const float density_z_next = output[z_next_base + 1];
            grad_z = (density_z_next - density_z_prev) / (2.0f * dz);
        }
    }

    // ---- r 方向差分 ----
    float grad_r = 0.0f;

    if (r_samples > 1) {
        if (r_idx == 0) {
            // Forward difference at inner radial boundary.
            const int r_next_base = ((phi_idx * z_samples + z_idx) * r_samples + (r_idx + 1)) * 4;
            const float density_center = output[center_base + 1];
            const float density_r_next = output[r_next_base + 1];
            grad_r = (density_r_next - density_center) / dr;
        } else if (r_idx == r_samples - 1) {
            // Backward difference at outer radial boundary.
            const int r_prev_base = ((phi_idx * z_samples + z_idx) * r_samples + (r_idx - 1)) * 4;
            const float density_center = output[center_base + 1];
            const float density_r_prev = output[r_prev_base + 1];
            grad_r = (density_center - density_r_prev) / dr;
        } else {
            // Central difference in the interior.
            const int r_prev_base = ((phi_idx * z_samples + z_idx) * r_samples + (r_idx - 1)) * 4;
            const int r_next_base = ((phi_idx * z_samples + z_idx) * r_samples + (r_idx + 1)) * 4;
            const float density_r_prev = output[r_prev_base + 1];
            const float density_r_next = output[r_next_base + 1];
            grad_r = (density_r_next - density_r_prev) / (2.0f * dr);
        }
    }

    // L1 范数，而不是 sqrt(grad_r^2 + grad_z^2 + grad_phi^2)。
    const float density_gradient = fabsf(grad_r) + fabsf(grad_z) + fabsf(grad_phi);

    // 第 3 通道写入密度梯度绝对值和。
    output[center_base + 3] = density_gradient;
}