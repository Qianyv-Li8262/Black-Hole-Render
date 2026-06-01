__device__ __forceinline__ float3 normalize(float3 v){
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x*inv_norm , v.y*inv_norm , v.z*inv_norm);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w +b.w);
}
__device__ __forceinline__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s,a.w * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float3 operator/(float3 a, float3 s) {
    return make_float3(a.x / s.x, a.y / s.y, a.z / s.z);
}
__device__ __forceinline__ float operator*(float3 s, float3 a) {
    return a.x * s.x + a.y * s.y + a.z * s.z;
}

__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float Luminance(float3 color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

__device__ __forceinline__ float karisWeight(float3 color) {
    return 1.0f / (Luminance(color) + 1.0f);
}

extern "C"
__global__ void tap13_downsample(
    cudaTextureObject_t img,
    float4* __restrict__ downsample_img,
    int width_original,int height_original,
    int width_output,int height_output
){
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_idx>=width_output || pixel_idy >= height_output) return;
    float u = (float)pixel_idx / (float)width_output;
    float v = (float)pixel_idy / (float)height_output;
    float half_step_x = 0.5f / width_original;
    float half_step_y = 0.5f / height_original;
    float3 g0 = make_float3(0.0f, 0.0f, 0.0f);// SW
    float3 g1 = make_float3(0.0f, 0.0f, 0.0f);// SE
    float3 g2 = make_float3(0.0f, 0.0f, 0.0f);// NW
    float3 g3 = make_float3(0.0f, 0.0f, 0.0f);// NE
    float3 g4 = make_float3(0.0f, 0.0f, 0.0f);// Ctr

    float4 tap;

    tap = tex2D<float4>(img, u - 2.0f * half_step_x, v - 2.0f * half_step_y);
    g2.x += tap.x; g2.y += tap.y; g2.z += tap.z;


    tap = tex2D<float4>(img, u, v - 2.0f * half_step_y);
    g2.x += tap.x; g2.y += tap.y; g2.z += tap.z;
    g3.x += tap.x; g3.y += tap.y; g3.z += tap.z;

    tap = tex2D<float4>(img, u + 2.0f * half_step_x, v - 2.0f * half_step_y);
    g3.x += tap.x; g3.y += tap.y; g3.z += tap.z;


    tap = tex2D<float4>(img, u - 1.0f * half_step_x, v - 1.0f * half_step_y);
    g4.x += tap.x; g4.y += tap.y; g4.z += tap.z;

    tap = tex2D<float4>(img, u + 1.0f * half_step_x, v - 1.0f * half_step_y);
    g4.x += tap.x; g4.y += tap.y; g4.z += tap.z;


    tap = tex2D<float4>(img, u - 2.0f * half_step_x, v);
    g0.x += tap.x; g0.y += tap.y; g0.z += tap.z;
    g2.x += tap.x; g2.y += tap.y; g2.z += tap.z;

    tap = tex2D<float4>(img, u, v);
    g0.x += tap.x; g0.y += tap.y; g0.z += tap.z;
    g1.x += tap.x; g1.y += tap.y; g1.z += tap.z;
    g2.x += tap.x; g2.y += tap.y; g2.z += tap.z;
    g3.x += tap.x; g3.y += tap.y; g3.z += tap.z;

    tap = tex2D<float4>(img, u + 2.0f * half_step_x, v);
    g1.x += tap.x; g1.y += tap.y; g1.z += tap.z;
    g3.x += tap.x; g3.y += tap.y; g3.z += tap.z;


    tap = tex2D<float4>(img, u - 1.0f * half_step_x, v + 1.0f * half_step_y);
    g4.x += tap.x; g4.y += tap.y; g4.z += tap.z;

    tap = tex2D<float4>(img, u + 1.0f * half_step_x, v + 1.0f * half_step_y);
    g4.x += tap.x; g4.y += tap.y; g4.z += tap.z;

    tap = tex2D<float4>(img, u - 2.0f * half_step_x, v + 2.0f * half_step_y);
    g0.x += tap.x; g0.y += tap.y; g0.z += tap.z;

    tap = tex2D<float4>(img, u, v + 2.0f * half_step_y);
    g0.x += tap.x; g0.y += tap.y; g0.z += tap.z;
    g1.x += tap.x; g1.y += tap.y; g1.z += tap.z;

    tap = tex2D<float4>(img, u + 2.0f * half_step_x, v + 2.0f * half_step_y);
    g1.x += tap.x; g1.y += tap.y; g1.z += tap.z;

    g0.x *= 0.25f; g0.y *= 0.25f; g0.z *= 0.25f;
    g1.x *= 0.25f; g1.y *= 0.25f; g1.z *= 0.25f;
    g2.x *= 0.25f; g2.y *= 0.25f; g2.z *= 0.25f;
    g3.x *= 0.25f; g3.y *= 0.25f; g3.z *= 0.25f;
    g4.x *= 0.25f; g4.y *= 0.25f; g4.z *= 0.25f;

    float w0 = karisWeight(g0);
    float w1 = karisWeight(g1);
    float w2 = karisWeight(g2);
    float w3 = karisWeight(g3);
    float w4 = karisWeight(g4);

    float w_sum = w0 + w1 + w2 + w3 + w4;
    float inv_w_sum = 1.0f / (w_sum + 1e-5f);
    int pid = pixel_idy * width_output + pixel_idx;
    downsample_img[pid].x = (g0.x * w0 * 0.125f + g1.x * w1 * 0.125f + g2.x * w2 * 0.125f + g3.x * w3 * 0.125f + g4.x * w4 * 0.500f) * inv_w_sum;
    downsample_img[pid].y = (g0.y * w0 * 0.125f + g1.y * w1 * 0.125f + g2.y * w2 * 0.125f + g3.y * w3 * 0.125f + g4.y * w4 * 0.500f) * inv_w_sum;
    downsample_img[pid].z = (g0.z * w0 * 0.125f + g1.z * w1 * 0.125f + g2.z * w2 * 0.125f + g3.z * w3 * 0.125f + g4.z * w4 * 0.500f) * inv_w_sum;
    downsample_img[pid].w = 1.0f;
}


extern "C"
__global__ void tent_upsampling_kernel1(
    cudaTextureObject_t img_small,
    float4* __restrict__ img_temp,
    int width_original,int height_original,
    int width_output,int height_output
){
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_idx>=width_output || pixel_idy >= height_output) return;
    float u = (float)pixel_idx / (float)width_output;
    float v = (float)pixel_idy / (float)height_output;
    float4 tex_get = tex2D<float4>(img_small,u,v);
    int pid = pixel_idy * width_output + pixel_idx;
    img_temp[pid] = tex_get;
}
extern "C" __global__ 
void tent_upsampling_kernel2(
    const float4* __restrict__ img_temp,
    float4* __restrict__ img_out,
    int ww, int hh
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * 32 + tx; // 32x32 = 1024 线性线程 ID

    int pixel_idx = blockIdx.x * 32 + tx;
    int pixel_idy = blockIdx.y * 32 + ty;

    __shared__ float4 halo[34][34];

    #pragma unroll
    for (int i = thread_id; i < 34 * 34; i += 1024) {
        int hy = i / 34;
        int hx = i % 34;

        int gx = blockIdx.x * 32 - 1 + hx;
        int gy = blockIdx.y * 32 - 1 + hy;

        gx = max(0, min(gx, ww - 1));
        gy = max(0, min(gy, hh - 1));

        halo[hy][hx] = img_temp[gy * ww + gx];
    }

    __syncthreads();

    if (pixel_idx >= ww || pixel_idy >= hh) return;

    float4 a = halo[ty][tx];         // Top-Left
    float4 b = halo[ty][tx + 1];     // Top
    float4 c = halo[ty][tx + 2];     // Top-Right
    float4 d = halo[ty + 1][tx];     // Left
    float4 e = halo[ty + 1][tx + 1]; // Center
    float4 f = halo[ty + 1][tx + 2]; // Right
    float4 g = halo[ty + 2][tx];     // Bottom-Left
    float4 h = halo[ty + 2][tx + 1]; // Bottom
    float4 i = halo[ty + 2][tx + 2]; // Bottom-Right

    float4 result;
    result.x = (a.x + c.x + g.x + i.x) * 0.0625f + (b.x + d.x + f.x + h.x) * 0.125f + e.x * 0.25f;
    result.y = (a.y + c.y + g.y + i.y) * 0.0625f + (b.y + d.y + f.y + h.y) * 0.125f + e.y * 0.25f;
    result.z = (a.z + c.z + g.z + i.z) * 0.0625f + (b.z + d.z + f.z + h.z) * 0.125f + e.z * 0.25f;
    result.w = (a.w + c.w + g.w + i.w) * 0.0625f + (b.w + d.w + f.w + h.w) * 0.125f + e.w * 0.25f;

    // 6. 写入高分辨率输出图像
    img_out[pixel_idy * ww + pixel_idx] = result;
}


// 经典的 ACES Filmic Tonemapping 拟合公式 (让高光优雅弯曲，防止死白)
__device__ __forceinline__ float3 ACESTonemap(float3 color) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    // 逐分量手动计算，不依赖任何 float3 运算符重载
    float res_x = (color.x * (a * color.x + b)) / (color.x * (c * color.x + d) + e);
    float res_y = (color.y * (a * color.y + b)) / (color.y * (c * color.y + d) + e);
    float res_z = (color.z * (a * color.z + b)) / (color.z * (c * color.z + d) + e);

    // 使用标准 CUDA 标量 math 函数进行 Clamp
    res_x = fmaxf(0.0f, fminf(res_x, 1.0f));
    res_y = fmaxf(0.0f, fminf(res_y, 1.0f));
    res_z = fmaxf(0.0f, fminf(res_z, 1.0f));

    return make_float3(res_x, res_y, res_z);
}

__device__ __forceinline__ float screen_dither(int x, int y) {
    float h = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
    return h - floorf(h);
}

extern "C" __global__
void combine_hdr_bloom_kernel(
    const float4* __restrict__ img_original, // 原始 Full-Resolution 渲染图 (HDR)
    cudaTextureObject_t tex_bloom,          // 最终计算好的 1/2 分辨率平滑光晕图 (U1, 开启 Bilinear 采样器)
    unsigned char* __restrict__ img_final,   // 最终输出到屏幕的贴图 (uint8 RGBA)
    int w, int h,                            // 原始分辨率宽度和高度
    float intensity                          // 外部传入的可调 Intensity 参数 (推荐 1.5 - 2.5)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int pid = y * w + x;
    float u = (float)x / (float)w;
    float v = (float)y / (float)h;

    // 1. 读取原始锐利的黑洞渲染图像 (HDR 空间)
    float4 orig_pixel = img_original[pid];
    float3 color_hdr = make_float3(orig_pixel.x, orig_pixel.y, orig_pixel.z);

    // 2. 利用 GPU 硬件双线性插值采样 1/2 分辨率的光晕图 U1
    float4 bloom_pixel = tex2D<float4>(tex_bloom, u, v);
    float3 bloom_hdr = make_float3(bloom_pixel.x, bloom_pixel.y, bloom_pixel.z);

    // 3. 线性相加 (Additive Blend) 融入光晕
    color_hdr.x += bloom_hdr.x * intensity;
    color_hdr.y += bloom_hdr.y * intensity;
    color_hdr.z += bloom_hdr.z * intensity;

    // 4. ACES 色调映射：把暴力的物理 HDR 光强优雅地收拢到屏幕能显示的 [0.0, 1.0] 范围内
    float3 color_ldr = ACESTonemap(color_hdr);

    // 5. Gamma 2.2 校正
    float r_val = __powf(color_ldr.x, 0.4545f) * 255.0f;
    float g_val = __powf(color_ldr.y, 0.4545f) * 255.0f;
    float b_val = __powf(color_ldr.z, 0.4545f) * 255.0f;

    // 6. 屏幕空间抖动消除色带
    float dither = screen_dither(x, y) * 3.0f;

    // 7. 写入 uint8 RGBA 输出
    int out_idx = pid * 4;
    img_final[out_idx + 0] = (unsigned char)fmaxf(0.0f, fminf(r_val + dither, 255.0f));
    img_final[out_idx + 1] = (unsigned char)fmaxf(0.0f, fminf(g_val + dither, 255.0f));
    img_final[out_idx + 2] = (unsigned char)fmaxf(0.0f, fminf(b_val + dither, 255.0f));
    img_final[out_idx + 3] = 255;
}