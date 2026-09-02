# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
import numpy as np
import cupy as cp
import matplotlib.pyplot as plot
import matplotlib.image as image
import os
generation_source=r'''
__device__ __forceinline__ float fract(float x) { 
    return x - floorf(x); 
}


__device__ float hash11(float n) {
    return fract(sinf(n) * 43758.5453123f);
}


__device__ float noise1D(float x) {
    float i = floorf(x);
    float f = fract(x);
    

    float u = f * f * (3.0f - 2.0f * f);
    

    return hash11(i) * (1.0f - u) + hash11(i + 1.0f) * u;
}


__device__ float fbm_sharp(float r) {
    float v = 0.0f;
    float amp = 1.0f;     
    float freq = 1.0f;    
    float max_amp = 0.0f; 
    
    // 叠加 5 阶细节 (Octaves)
    for(int i = 0; i < 5; i++) {
        // 1. 获取平滑噪声，并重新映射到 [-1.0, 1.0]
        float n = noise1D(r * freq) * 2.0f - 1.0f;
        
        // 2. 绝对值制造“锐利折角”，翻转制造“亮光尖峰” (Ridged 操作)
        n = 1.0f - fabsf(n); 
        
        // 3. 可选：平方操作，让亮的环变窄变锐利，暗的缝隙变宽
        n = n * n; 
        
        // 累加到总值
        v += n * amp;
        
        // 记录此时可能达到的最大值
        max_amp += amp; 
        
        // 频率翻倍 (Lacunarity)，振幅减半 (Gain)
        freq *= 2.0f;  
        amp *= 0.5f;   
    }
    
    // 4. 完美归一化：除以理论最大值，强行锁死在 [0.0, 1.0]
    return v / max_amp; 
}
__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = __saturatef((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}


// 黑体辐射颜色近似（温度 -> RGB）
__device__ float3 temperature_to_color(float temp) {
    // temp: 温度（开尔文），范围 1000K - 40000K
    // 修正版 Tanner Helland 算法
    
    // 注意：原算法的拟合公式是基于 temp / 100.0f 计算的！
    float t = temp / 100.0f; 
    float r, g, b;
    
    // --- Red ---
    if (t <= 66.0f) {
        r = 1.0f;
    } else {
        r = 1.292936186f * powf(t - 60.0f, -0.1332047592f);
        r = fminf(1.0f, fmaxf(0.0f, r));
    }
    
    // --- Green ---
    if (t <= 66.0f) {
        g = 0.39008157876f * logf(t) - 0.631841444f;
        g = fminf(1.0f, fmaxf(0.0f, g));
    } else {
        g = 1.129890861f * powf(t - 60.0f, -0.0755148492f);
        g = fminf(1.0f, fmaxf(0.0f, g));
    }
    
    // --- Blue ---
    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) { // 原算法中低于 1900K 蓝光为 0
        b = 0.0f;
    } else {
        b = 0.543206789f * logf(t - 10.0f) - 1.196254089f;
        b = fminf(1.0f, fmaxf(0.0f, b));
    }
            r = fminf(1.0f, fmaxf(0.0f, r));
    g = fminf(1.0f, fmaxf(0.0f, g));
    b = fminf(1.0f, fmaxf(0.0f, b));
    return make_float3(r, g, b);
}






// 吸积盘密度函数
__device__ float disk_density(float posz, float r_disk) {
    // 参数说明：
    // pos: 当前位置
    // r_disk: 到旋转轴的距离 sqrt(x^2 + y^2)
    

    float z_scale = 0.18f;  // 厚度参数，越小盘越薄
    float vertical_density = expf(-fabsf(posz) / z_scale);
    
 
    float radial_density = powf((r_disk-6.349489f) / 0.1f, -0.7f);

    //float sharp_noise = fbm_sharp((r_disk-6.349489f) * 1.5f); 
    float sharp_noise = 1.0f;
    float rings = powf(sharp_noise, 2.0f); 


    // 组合所有因素
    float density = vertical_density * radial_density * rings;
    
    // 密度缩放因子（控制整体不透明度）
    density *= 1.2f;
    
    return fmaxf(0.0f, density);
}


__device__ float disk_temperature(float r_disk) {


    
    //float T0 = 130000.0f;
    //float xi = r_disk + 1.0f + 0.25f/r_disk; // 转换为标准坐标比例
    //float q_factor = (1.0f - sqrt(6.0f/xi)) / (1.0f - 3.0f/xi);
    //float temp = T0 * pow(xi, -0.75f) * pow(q_factor, 0.25f);

    float T0 = 18400.0f;
    float r = r_disk * 1.0f;
    //float u = 1 / (2.0f*r);float invupl = 1.0f/(1.0f + u);
    //float temp = T0 *powf((r-4.9495f)/0.8f,-0.82f)*powf(1.0f-sqrtf(0.05f/(r-4.9495f)),1.5f);
    //float ttt = (1-2.4494897f*rsqrtf(r)*invupl)/r/r/r*invupl*invupl*invupl*invupl*invupl*invupl;
    //float temp = T0 * powf(ttt,0.25f);
    float x = r - 4.9495f;

    float base_temp = T0 * powf(x / 0.8f, -1.0f);

    float r_start = 0.05f;
    float width = 1.2f;   // 核心参数：控制从 0 升到最高温的物理宽度！
                      // 值越大，内圈最热的蓝色区域就越宽、越柔和。

    float envelope = smoothstep(r_start, r_start + width, x);


    float temp = base_temp * envelope;

    temp = fmaxf(500.0f, temp);
    // float temp=1500.0f;
    return temp;
}

// 计算吸积盘在某点的发射颜色和强度
__device__ float4 disk_emission(float3 pos, float r_disk) {
    // 1. 计算温度
    float temp = disk_temperature(r_disk);
    
    // 2. 温度转颜色
    float3 color = temperature_to_color(temp);
    
    // 3. 发射强度（内圈更亮）
    float intensity = 10.0f*powf(4/(r_disk-6.2494f),2.0f);
    // intensity = fminf(3.0f, intensity);
    intensity = fminf(20.0f, fmaxf(0.0f, intensity));

    
    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}

extern "C" __global__
void generateLutPhysics(
float* __restrict__ out_array,
int r_pixels,
int z_pixels,
float zmax,
float rmin,
float rmax

){

int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid >= r_pixels * z_pixels) return;

int r_pix = tid % r_pixels;
int z_pix = tid / r_pixels;
float r_phys = rmin + (rmax - rmin) * (float)r_pix / (float)r_pixels;
float z_phys = (float)z_pix/(float)z_pixels*zmax;
float density = disk_density(z_phys,r_phys);
float temp = disk_temperature(r_phys);
float intensity =  30.0f*powf(4/(r_phys-4.0f),2.0f);
//float intensity = 0.3f * (temp/1000)* (temp/1000)* (temp/1000)* (temp/1000);
intensity = fminf(2000.0f, fmaxf(0.0f, intensity));
int u = tid *4;
out_array[u+0]=density;
out_array[u+1]=temp;
out_array[u+2]=intensity;
out_array[u+3]=0.0f;

}

'''


array=cp.empty((50,1500,4),dtype=cp.float32)
kernel = cp.RawKernel(generation_source, 'generateLutPhysics')
kernel((293,),(256,),(array,cp.int32(1500),cp.int32(50),cp.float32(2.5),cp.float32(4.9495),cp.float32(35.0)))
np_arr = array.get()
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(base_path, 'cache')
os.makedirs(cache_dir, exist_ok=True)
np.save(os.path.join(cache_dir, 'disk_lut_for_mov_disk.npy'), np_arr)
print("LUT 生成完成，形状:", np_arr.shape)
print(np_arr[5:10,110:120,:])
