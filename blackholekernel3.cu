// #include<cooperative_groups.h>
// #include <cuda_pipeline_primitives.h>

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

__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s,a.w * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}


__device__ __forceinline__ float tdot(float r){
return (2.0f*r+1.0f)/sqrtf(1.0f+4.0f*r*r-8.0f*r);
}

__device__ __forceinline__ float phidot(float r){
float a = r*sqrtf(r);
float b = sqrtf(1.0f+4.0f*r*r-8.0f*r);
return 8.0f*a/b/(2.0f*r+1.0f)/(2.0f*r+1.0f);
}

__device__ __forceinline__ float rand_float(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)seed / 4294967296.0f; // 归一化到 [0, 1)
}


__device__ __forceinline__ float fractf(float x) {
    return x - floorf(x);
}


__device__ float hash31(float x, float y, float z) {
    float3 p3 = make_float3(x, y, z);
    p3.x = fractf(p3.x * 0.1031f);
    p3.y = fractf(p3.y * 0.1030f);
    p3.z = fractf(p3.z * 0.0973f);
    float dot_val = p3.x * (p3.y + 33.33f) + p3.y * (p3.z + 33.33f) + p3.z * (p3.x + 33.33f);
    return fractf((p3.x + p3.y + p3.z) * dot_val);
}


__device__ float3 procedural_stars(float3 dir, int frames) {
    float3 total_stars = make_float3(0.0f, 0.0f, 0.0f);
    

    float lod_blend = __expf(-(float)(frames - 1) * 0.5f); 
    
    float sharp1_target = 25.0f; // 静止时极度尖锐
    float sharp1_moving = 2.0f;  // 移动时极度模糊（扩散为大光斑防闪烁）
    // 根据运动状态插值当前的锐度
    float s1 = sharp1_moving * lod_blend + sharp1_target * (1.0f - lod_blend);
    // 【能量守恒定律】：面积变大了，亮度必须按等比例降低，否则满屏白光
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
        float dist2 = dx*dx + dy*dy + dz*dz;
        
        // 应用动态模糊
        float star_shape = __expf(-dist2 * s1); 
        // 亮度乘以能量守恒系数
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
        float dist2 = dx*dx + dy*dy + dz*dz;
        
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


__device__ float4 disk_emission(float temp,float intensity,cudaTextureObject_t lut_color) {


    float4 color = tex2D<float4>(lut_color,(temp-1000.0f)/20000.0f,0.5f);
    


    
    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}

extern "C" __global__
void blackholekernel(
float4* __restrict__ raw_img,
cudaTextureObject_t tex_obj,
cudaTextureObject_t lut_physics,
cudaTextureObject_t lut_color,
const float cam_pos_x,
const float cam_pos_y,
const float cam_pos_z,
const float fwd_x,
const float fwd_y,
const float fwd_z,
const float right_x,
const float right_y,
const float right_z,
const float up_x,
const float up_y,
const float up_z,
const int imgwidth,const int imgheight,
const float physwidth,const float physheight,
const float focal_length,const float step,const int maxstep,const int jitternum,const int frames

){




float3 fwd = make_float3(fwd_x,fwd_y,fwd_z);
float3 right = make_float3(right_x,right_y,right_z);
float3 up = make_float3(up_x,up_y,up_z);



int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

if( pixel_idx >= imgwidth || pixel_idy >= imgheight ) return;
float4 buffer=make_float4(0.0f,0.0f,0.0f,0.0f);


float jitterx;
float jittery;
float physical_x;
float physical_y;
// for(int i = 0;i < jitternum;++i){

jitterx = rand_float((unsigned int)pixel_idx+frames);
jittery = rand_float((unsigned int)pixel_idy+12345+frames);
physical_x = (((float)pixel_idx+jitterx)/(float)imgwidth - 0.5f) * physwidth;
physical_y = (((float)pixel_idy+jittery)/(float)imgheight - 0.5f) * physheight;
float3 cam_pos=make_float3(cam_pos_x,cam_pos_y,cam_pos_z);
float r = length(cam_pos);
float3 d = make_float3(
    fwd.x * focal_length - right.x * physical_x - up.x * physical_y,
    fwd.y * focal_length - right.y * physical_x - up.y * physical_y,
    fwd.z * focal_length - right.z * physical_x - up.z * physical_y
    );
// 出射单位向量
d=normalize(d);
float u=1.0f/(2.0f * r);
float upl = 1.0f+u;
float umi = 1.0f-u;
float factor = upl/umi;
float n=upl*upl*upl/umi;
float3 p = d * n;
float lz = cam_pos.x*p.y-cam_pos.y*p.x;
bool flag = true;
float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
// bool hit_disk=false;


for (int s = 0 ; s < maxstep && flag ; ++s){
    float3 prev_pos = cam_pos;

    //step 1

float rmhalf = r-0.5f;
float g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
float uplsq=upl*upl;
float uu=1.0f/(uplsq*uplsq);
float3 k11 = p * uu;
float3 k12 = g * cam_pos;

//自适应步长

// float current_step = step * fminf(10.0f, fmaxf(0.05f, (r - 0.95f))); 



// if (r > 1.2f && r < 18.0f && fabsf(cam_pos.z) < 2.0f) {

//     float z_factor = fabsf(cam_pos.z) / 2.0f;
    

//     float multiplier = 0.05f + 0.15f * (z_factor * z_factor); 
    
//     current_step *= multiplier;
// }

bool in_disk_volume = (r > 4.8f && r < 18.0f && fabsf(cam_pos.z) < 2.0f); 
// in_disk_volume 为 true 时(1.0)，应用 0.05f，否则为 1.0f
float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
float current_step = step * fminf(50.0f, fmaxf(0.05f, r - 0.54f)) * zone_multiplier;
// current_step = r<1.89?0.1*current_step:current_step;
// if (r > 1.4f && r < 17.0f && fabsf(cam_pos.z) < 0.7f){
//     current_step *=0.05f;
// }

//step 2
float stephalf = current_step*0.5f;
float3 pos_tmp=cam_pos+(stephalf)*k11;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
rmhalf = r-0.5f;
g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
uplsq=upl*upl;
uu=1.0f/(uplsq*uplsq);
float3 k21 = (p+(stephalf)*k12)*uu;
float3 k22 = pos_tmp * g;

cam_pos = cam_pos+(current_step)*(k21);
p = p+(current_step)*(k22);
r = length(cam_pos);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
float3 temp = make_float3((cam_pos.x+prev_pos.x)/2.0f,(cam_pos.y+prev_pos.y)/2.0f,0.0f);
// float r_disk = sqrtf(cam_pos.x * cam_pos.x + cam_pos.y * cam_pos.y);
float r_disk_sq = temp.x * temp.x + temp.y * temp.y;
bool indisk = (r_disk_sq > 24.4974f && r_disk_sq < 272.25f && fabsf(cam_pos.z) < 1.5f);


if (indisk) {
float r_disk=sqrtf(r_disk_sq);
    float4 parameters = tex2D<float4>(lut_physics,(r_disk-4.9495f)/11.5505f,fabsf(cam_pos.z)/1.5f);
    
    // calculate g factor
    float td = tdot(r);
    float pd = phidot(r);
    float g = factor /(td+pd*lz);
    // g = 1.0f;

    float4 emission = disk_emission(fmaxf(parameters.y*g,1000.0f),parameters.z*g*g*g*g,lut_color);
    // float4 emission = disk_emission(19000,parameters.z*g*g*g*g,lut_color);
    
    float ravg = (length(prev_pos)+r)/2.0f;
    float uuu=1.0f+1.0f/(2.0f*ravg);

    float k = 2.0f; 
    float intensity_factor = 1.0f - __expf(-(k * parameters.z)*(k * parameters.z));
    float step_opacity = parameters.x * 1.7f*uuu*uuu*length(cam_pos-prev_pos)* intensity_factor / g;
    step_opacity = parameters.y*g<1500.0f ? 0.0f : fminf(step_opacity, 1.0f);
    // step_opacity = fminf(step_opacity, 1.0f);
    

    float transmittance = 1.0f - accumulated_color.w;
    accumulated_color.x += emission.x * step_opacity * transmittance;
    accumulated_color.y += emission.y * step_opacity * transmittance;
    accumulated_color.z += emission.z * step_opacity * transmittance;
    accumulated_color.w += step_opacity * transmittance;
    // 调试：直接显示 g 因子
// accumulated_color.x = (g>1)?0:1;
// accumulated_color.y = 0.0f;
// accumulated_color.z = 0.0f;


    if (accumulated_color.w > 0.99f) {
        flag = false;
    }
}




// 终止条件：掉入黑洞、飞出边界、或数值异常
if(r<0.55f || r>140.0f ) {flag = false;}
}

float4 color;
if (r >=0.55f && !isnan(r)) {



float3 final_dir = normalize(p);

    float phi = atan2f(final_dir.y, -final_dir.x); 
    float theta = asinf(-final_dir.z);

    float tex_u = phi*0.1591549f+0.5f;
    float tex_v = theta* 0.3183099f+0.5f;
    


    float4 bkgd = tex2D<float4>(tex_obj, tex_u, tex_v);

    
    
// 注释掉以下这段以禁用程序化星空生成
    
// bkgd.x *= 0.6f;
// bkgd.y *= 0.6f;
// bkgd.z *= 0.6f;

// float3 p_stars = procedural_stars(final_dir,frames);
// bkgd.x += p_stars.x;
// bkgd.y += p_stars.y;
// bkgd.z += p_stars.z;

// 注释掉以上这段以禁用程序化星空生成





color = accumulated_color + bkgd * (1.0f - accumulated_color.w);



} else {
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 0] = 0.0f;
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 1] = 0.0f;
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 2] = 0.0f;
color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
}

buffer = buffer+color;
// }
// buffer = buffer*(1.0f/(float)jitternum);
    int pixel_index = (pixel_idy * imgwidth + pixel_idx);
    // raw_img[pixel_index + 0] = buffer.x; // R
    // raw_img[pixel_index + 1] = buffer.y; // G
    // raw_img[pixel_index + 2] = buffer.z; // B
    // raw_img[pixel_index + 3] = 0.0; 
    raw_img[pixel_index] = make_float4(buffer.x,buffer.y,buffer.z,0.0);
}
