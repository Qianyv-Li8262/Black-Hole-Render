__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w +b.w);
}
__device__ __forceinline__ float4 operator/(float4 a, float s) {
    return make_float4(a.x / s, a.y / s, a.z / s,a.w/s);
}
__device__ __forceinline__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s,a.w * s);
}
extern "C"
__global__ void gaussianBlurH(float4* __restrict__ out, int width, int height, 
                              cudaTextureObject_t tex,float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / (float)width;
    float v = (y + 0.5f) / (float)height;

    const float w[5] = {0.19638062f, 0.29675293f, 0.09442139f, 
                        0.01037598f, 0.00025940f};
    const float off[5] = {0.0f, 1.41176471f, 3.29411765f, 
                          5.17647059f, 7.05882353f};

    float4 sum = tex2D<float4>(tex, u, v) * w[0];
    for (int i = 1; i < 5; i++) {
        float du = (off[i] * scale) / width;
        sum =sum+ tex2D<float4>(tex, u + du, v) * w[i];
        sum =sum+ tex2D<float4>(tex, u - du, v) * w[i];
    }
    out[y * width + x] = sum;
}


extern "C"
__global__ void gaussianBlurW(float4* __restrict__ out, int width, int height, 
                              cudaTextureObject_t tex,float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    const float w[5] = {0.19638062f, 0.29675293f, 0.09442139f, 
                        0.01037598f, 0.00025940f};
    const float off[5] = {0.0f, 1.41176471f, 3.29411765f, 
                          5.17647059f, 7.05882353f};

    float4 sum = tex2D<float4>(tex, u, v) * w[0];
    for (int i = 1; i < 5; i++) {
        float dv = (off[i] * scale) / height;
        sum =sum+ tex2D<float4>(tex, u, v + dv) * w[i];
        sum =sum+ tex2D<float4>(tex, u, v - dv) * w[i];
    }
    out[y * width + x] = sum;
}



__device__ float4 bicubicSample(
    cudaTextureObject_t tex,
    float2 uv,          // 归一化坐标
    float texWidth,
    float texHeight
) {
    // 1. 转换到基于连续像素的坐标系
    float px = uv.x * texWidth - 0.5f;
    float py = uv.y * texHeight - 0.5f;

    // 2. 提取整数和小数部分
    float ix = floorf(px);
    float iy = floorf(py);
    float fx = px - ix;
    float fy = py - iy;

    // 3. 计算 B-Spline 三次样条权重 (严格保证能量守恒，无负数振铃)
    float fx2 = fx * fx;
    float fx3 = fx2 * fx;
    float w0x = (1.0f - fx) * (1.0f - fx) * (1.0f - fx) / 6.0f;
    float w1x = (3.0f * fx3 - 6.0f * fx2 + 4.0f) / 6.0f;
    float w2x = (-3.0f * fx3 + 3.0f * fx2 + 3.0f * fx + 1.0f) / 6.0f;
    float w3x = fx3 / 6.0f;

    float fy2 = fy * fy;
    float fy3 = fy2 * fy;
    float w0y = (1.0f - fy) * (1.0f - fy) * (1.0f - fy) / 6.0f;
    float w1y = (3.0f * fy3 - 6.0f * fy2 + 4.0f) / 6.0f;
    float w2y = (-3.0f * fy3 + 3.0f * fy2 + 3.0f * fy + 1.0f) / 6.0f;
    float w3y = fy3 / 6.0f;

    // 4. 利用硬件双线性插值特性的权重合并
    float g0x = w0x + w1x;
    float g1x = w2x + w3x;
    float h0x = (w1x / g0x) - 0.5f;
    float h1x = (w3x / g1x) + 1.5f;

    float g0y = w0y + w1y;
    float g1y = w2y + w3y;
    float h0y = (w1y / g0y) - 0.5f;
    float h1y = (w3y / g1y) + 1.5f;

    // 5. 计算最终的 4 个采样点归一化坐标 (去除了多余的 + 0.5f 偏移！)
    float u0 = (ix + h0x + 0.5f) / texWidth;
    float u1 = (ix + h1x + 0.5f) / texWidth;
    float v0 = (iy + h0y + 0.5f) / texHeight;
    float v1 = (iy + h1y + 0.5f) / texHeight;

    // 6. 进行 4 次硬件双线性纹理采样
    float4 t00 = tex2D<float4>(tex, u0, v0);
    float4 t10 = tex2D<float4>(tex, u1, v0);
    float4 t01 = tex2D<float4>(tex, u0, v1);
    float4 t11 = tex2D<float4>(tex, u1, v1);

    // 7. 手动完成最后的权重组合
    float4 ty0 = t00 * g0x + t10 * g1x;
    float4 ty1 = t01 * g0x + t11 * g1x;

    return ty0 * g0y + ty1 * g1y;
}


extern "C" __global__
void compositeBloom(
    uchar4* __restrict__ output,
    int outWidth, int outHeight,
    cudaTextureObject_t originalTex,
    cudaTextureObject_t* bloomTextures,
    int num_levels,
    int originalWidth, int originalHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outWidth || y >= outHeight) return;

    float u = (float)(x + 0.5f) / (float)outWidth;
    float v = (float)(y + 0.5f) / (float)outHeight;

    // 1. 原始颜色
    float4 color = tex2D<float4>(originalTex, u, v);

    // 2. Bloom 累加（分量展开）
    const float bloomWeights[8] = {1.0f, 1.5f, 1.0f, 1.5f, 1.8f, 1.0f, 1.0f, 1.0f};
    float4 bloom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int oct = 0; oct < num_levels; ++oct) {
        float scale = (float)(1 << (oct + 1));   // 2^(oct+1)
        float bw = (float)(originalWidth / scale);
        float bh = (float)(originalHeight / scale);
        float bu = u;   // 映射到该级纹理的 UV
        float bv = v;

        float4 bs = bicubicSample(bloomTextures[oct], make_float2(bu, bv), bw, bh);
        if (oct<=8){
            bloom.x += bs.x * bloomWeights[oct];
            bloom.y += bs.y * bloomWeights[oct];
            bloom.z += bs.z * bloomWeights[oct];
            bloom.w += bs.w * bloomWeights[oct];
        } else {
            bloom.x += bs.x;
            bloom.y += bs.y;
            bloom.z += bs.z;
            bloom.w += bs.w;
        }
    }
    color.x *= 0.55f;
    color.y *= 0.55f;
    color.z *= 0.55f;
    // 原始颜色叠加 Bloom（0.08 强度）
    color.x += bloom.x * 0.08f;
    color.y += bloom.y * 0.08f;
    color.z += bloom.z * 0.08f;
    color.w += bloom.w * 0.08f;

    // 3. 曝光提升
    color.x *= 0.15f;
    color.y *= 0.15f;
    color.z *= 0.15f;

    // 4. 改良 Reinhard 色调映射
    // 先 pow(color, 1.5)
    color.x = powf(color.x, 1.5f);
    color.y = powf(color.y, 1.5f);
    color.z = powf(color.z, 1.5f);

    // 除以 (1 + color)
    color.x = color.x / (1.0f + color.x);
    color.y = color.y / (1.0f + color.y);
    color.z = color.z / (1.0f + color.z);

    // 逆 pow(..., 1/1.5)
    color.x = powf(color.x, 1.0f / 1.5f);
    color.y = powf(color.y, 1.0f / 1.5f);
    color.z = powf(color.z, 1.0f / 1.5f);

    // 5. S 曲线对比度增强：mix(color, color*color*(3-2*color), 1.0)
    // 即 color = color * color * (3 - 2*color)
    float tmpR = color.x * color.x * (3.0f - 2.0f * color.x);
    float tmpG = color.y * color.y * (3.0f - 2.0f * color.y);
    float tmpB = color.z * color.z * (3.0f - 2.0f * color.z);
    color.x = tmpR;
    color.y = tmpG;
    color.z = tmpB;

    // 6. 通道偏色
    color.x = powf(color.x, 1.3f);
    color.y = powf(color.y, 1.20f);
    color.z = powf(color.z, 1.0f);

// 7. 真正的饱和度控制 (Saturation)
    // 根据 Rec.709 标准提取像素的感知亮度 (Luma)
    float luma = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
    
    // 饱和度参数 (1.0 = 不变, > 1.0 = 增加饱和度, 0.0 = 黑白)
    float saturation = 1.0f; // 建议在 1.2f ~ 1.6f 之间尝试
    
    // 公式: color = Luma + (color - Luma) * Saturation
    color.x = luma + (color.x - luma) * saturation;
    color.y = luma + (color.y - luma) * saturation;
    color.z = luma + (color.z - luma) * saturation;

    // 防止拉伸后出现负数或溢出，最后进行严格裁剪 (Clamp)
    color.x = fmaxf(0.0f, fminf(color.x * 1.01f, 1.0f));
    color.y = fmaxf(0.0f, fminf(color.y * 1.01f, 1.0f));
    color.z = fmaxf(0.0f, fminf(color.z * 1.01f, 1.0f));

    // 8. Gamma 校正
    float gamma = 0.7f / 2.2f;
    color.x = powf(color.x, gamma)* 255.0f;
    color.y = powf(color.y, gamma)* 255.0f;
    color.z = powf(color.z, gamma)* 255.0f;
    output[y * outWidth + x] = make_uchar4((unsigned char)color.x,(unsigned char)color.y,(unsigned char)color.z,255);
}



extern "C" __global__
void extractBright(
    const float4* __restrict__ accum,
    float4* __restrict__ bright_out,
    int w, int h, float frames, float threshold
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int pid = y * w + x;
    int c_idx = pid;
    
    float r = accum[c_idx].x / frames;
    float g = accum[c_idx].y / frames;
    float b = accum[c_idx].z / frames;

    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float knee = 0.5f; 
    float soft = luma - threshold + knee;
    soft = fmaxf(0.0f, fminf(soft, 2.0f * knee));
    soft = soft * soft / (4.0f * knee + 0.0001f);
    
    float weight = fmaxf(soft, luma - threshold) / fmaxf(luma, 0.0001f);
    
    weight = fminf(1.0f, weight);

    bright_out[c_idx]   = make_float4(r*weight,g*weight,b*weight,1.0f);
}



extern "C" __global__
void downsample2x(
    cudaTextureObject_t input_tex, 
    float4* __restrict__ output_img,
    int out_w, int out_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    // 手动获取高分辨率贴图上的 4 个像素中心点 (安全降采样)
    float dx = 1.0f / (out_w * 2.0f);
    float dy = 1.0f / (out_h * 2.0f);
    float u = (x * 2.0f + 0.5f) * dx;
    float v = (y * 2.0f + 0.5f) * dy;

    float4 c00 = tex2D<float4>(input_tex, u, v);
    float4 c10 = tex2D<float4>(input_tex, u + dx, v);
    float4 c01 = tex2D<float4>(input_tex, u, v + dy);
    float4 c11 = tex2D<float4>(input_tex, u + dx, v + dy);

    output_img[y * out_w + x] = (c00 + c10 + c01 + c11) * 0.25f;
}