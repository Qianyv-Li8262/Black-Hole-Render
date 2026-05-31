// #include<cooperative_groups.h>
// #include <cuda_pipeline_primitives.h>
// #include "cuda_noise.cuh"  // 不再需要 —— 噪声已预烘焙到 3D 纹理

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
__device__ __forceinline__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float operator*(float3 s, float3 a) {
    return a.x * s.x + a.y * s.y + a.z * s.z;
}
__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 boost(float3 b, float3 vec, float gamma){
    float len2 = b.x * b.x + b.y * b.y + b.z * b.z;
    if (len2 < 1e-12f) {
        return vec;
    }
    float3 bnor = b * rsqrtf(len2);
    float3 vv = vec + (gamma - 1.0f) * (bnor * vec) * bnor;
    return vv;
}

__device__ __forceinline__ float tdot(float r){
return (2.0f*r+1.0f)/sqrtf(1.0f+4.0f*r*r-8.0f*r);
}

__device__ __forceinline__ float phidot(float r){
float a = r*sqrtf(r);
float b = sqrtf(1.0f+4.0f*r*r-8.0f*r);
return 8.0f*a/b/(2.0f*r+1.0f)/(2.0f*r+1.0f);
}

__device__ __forceinline__ unsigned int pcg_hash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand_float(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)seed / 4294967296.0f;
}

__device__ __forceinline__ float2 hammersley(int i, int jitternum, unsigned int pixel_idx, unsigned int pixel_idy, unsigned int frames) {
    float h_x = (float)i / (float)jitternum;
    unsigned int reversed = __brev(i);
    float h_y = (float)reversed * 2.3283064365386963e-10f;

    float shift_x = rand_float(pixel_idx^pcg_hash(pixel_idy) ^( frames * 1919810));
    float shift_y = rand_float(pixel_idy^pcg_hash(pixel_idx*114514) ^( frames * 1919810));

    float j_x = h_x + shift_x;
    j_x = j_x - floorf(j_x);

    float j_y = h_y + shift_y;
    j_y = j_y - floorf(j_y);

    return make_float2(j_x, j_y);
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
        float dist2 = dx*dx + dy*dy + dz*dz;

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


    float4 color = tex2D<float4>(lut_color,(temp-510.0f)/20000.0f,0.5f);




    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}
__device__ __forceinline__ void tdpd(float r, float *td, float *pd) {
    float r2 = r * r;
    float sqrt_term = sqrtf(1.0f + 4.0f * r2 - 8.0f * r);
    float denom = 2.0f * r + 1.0f;

    *td = denom / sqrt_term;

    float a = r * sqrtf(r);
    *pd = 8.0f * a / (sqrt_term * denom * denom);
}
extern "C" __global__
void blackholekernel(
float4* __restrict__ raw_img,
cudaTextureObject_t tex_obj,
cudaTextureObject_t prebaked_disk,  // 预烘焙 3D 纹理: (r_disk, |z|, phi) -> (density, temp, intensity)
cudaTextureObject_t lut_color,
const float time,
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
const float vfwd,
const float vright,
const float vup,
const int imgwidth,const int imgheight,
const float physwidth,const float physheight,
const float focal_length,const float step,const int maxstep,const int jitternum,const int frames

){




float3 fwd = make_float3(fwd_x,fwd_y,fwd_z);
float3 right = make_float3(right_x,right_y,right_z);
float3 up = make_float3(up_x,up_y,up_z);

float3 beta = make_float3(vfwd,vright,vup);
float gamma = rsqrtf(1-(beta.x * beta.x + beta.y * beta.y + beta.z * beta.z));
int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

if( pixel_idx >= imgwidth || pixel_idy >= imgheight ) return;
float4 buffer=make_float4(0.0f,0.0f,0.0f,0.0f);


float jitterx;
float jittery;
float physical_x;
float physical_y;
for(int i = 0;i < jitternum;++i){

float2 jit = hammersley(i, jitternum, (unsigned int)pixel_idx, (unsigned int)pixel_idy,1);
jitterx = jit.x;
jittery = jit.y;
physical_x = (((float)pixel_idx+jitterx)/(float)imgwidth - 0.5f) * physwidth;
physical_y = (((float)pixel_idy+jittery)/(float)imgheight - 0.5f) * physheight;
float3 cam_pos=make_float3(cam_pos_x,cam_pos_y,cam_pos_z);


float r_pixel = sqrtf(physical_x * physical_x + physical_y * physical_y);
float theta = r_pixel / focal_length;
float sin_theta, cos_theta;
sincosf(theta, &sin_theta, &cos_theta);
float c_phi = 0.0f;
float s_phi = 0.0f;
if (r_pixel > 1e-6f) {
    c_phi = physical_x / r_pixel;
    s_phi = physical_y / r_pixel;
}
float3 tmp1 = make_float3(
  cos_theta,
(sin_theta * c_phi),
-(sin_theta * s_phi)
);


unsigned int depth_seed = pcg_hash(pixel_idx ^ pcg_hash(pixel_idy ^ pcg_hash(i ^ pcg_hash(frames))));
float depth_jitter = (float)depth_seed / 4294967296.0f;


float r = length(cam_pos);float u = 1.0f/(2.0f*r);
float upl = 1.0f+u;
float umi = 1.0f-u;
float factor = upl/umi;
float n=upl*upl*upl/umi;
float3 beta_global = beta.x*fwd + beta.y*right + beta.z*up;
float3 e0 = beta_global*gamma/(upl*upl);
float3 e1_fwd = boost(beta,make_float3(1.0f,0.0f,0.0f),gamma);
e1_fwd = (e1_fwd.x*fwd+e1_fwd.y*right+e1_fwd.z*up)/(upl*upl);
float3 e2_right = boost(beta,make_float3(0.0f,1.0f,0.0f),gamma);
e2_right = (e2_right.x*fwd+e2_right.y*right+e2_right.z*up)/(upl*upl);
float3 e3_up = boost(beta,make_float3(0.0f,0.0f,1.0f),gamma);
e3_up = (e3_up.x*fwd+e3_up.y*right+e3_up.z*up)/(upl*upl);
float3 d = normalize(tmp1.x*e1_fwd+tmp1.y*e2_right+tmp1.z*e3_up-1.0f*e0);
cam_pos = cam_pos + d * (depth_jitter *r/10.0f * step);
float3 p = d * n;
float3 p_init = p;
float lz = cam_pos.x*p.y-cam_pos.y*p.x;
bool flag = true;
float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);


for (int s = 0 ; s < maxstep && flag ; ++s){
    float3 prev_pos = cam_pos;

    //step 1

float rmhalf = r-0.5f;
float g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
float uplsq=upl*upl;
float uu=1.0f/(uplsq*uplsq);
float3 k11 = p * uu;
float3 k12 = g * cam_pos;



bool in_disk_volume = (r > 4.5f && r < 27.0f && fabsf(cam_pos.z) < 3.0f);
float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
float current_step = step * fminf(50.0f, fmaxf(0.005f, r - 0.54f)) * zone_multiplier;

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
float r_disk_sq = temp.x * temp.x + temp.y * temp.y;
bool indisk = (r_disk_sq > 24.4974f && r_disk_sq < 625.0f && fabsf(cam_pos.z) < 2.5f);


if (indisk) {
    float r_disk=sqrtf(r_disk_sq);

    // ---- g 因子（必须运行时计算，取决于每射线不同的度规） ----
    float td, pd; tdpd(r_disk, &td, &pd);

    float g = fmaxf((fabsf((factor*gamma+p_init*e0)/(td-pd*lz))-1.0f)*1.0f+1.0f, 0.01f);
    float rot = fmodf(pd * time / td, 6.283185307179586f);

    // phi_final = phi0 + rot — 先 atan2f 再加 rot，比和差角公式少一条 sincosf + 一条 atan2f
    float phi_final = atan2f(temp.y, temp.x) + rot;
    phi_final = fmodf(phi_final, 6.283185307f);
    if (phi_final < 0.0f) phi_final += 6.283185307f;

    // ---- 预烘焙 3D 纹理查表（替代原来的 lut_physics + 程序化噪声） ----
    // 3D 纹理 extent = (width=PHI, height=Z, depth=R)，对应 numpy C-order (R, Z, PHI, 4)
    // tex3D 坐标 (x, y, z) → (width, height, depth) → (phi, z, r_disk)
    //    x: phi:    0.0 .. 2π    → phi * 0.15915494
    //    y: |z|:    0.0 .. 2.5   → |z| / 2.5
    //    z: r_disk: 4.9495 .. 25 → (r_disk - 4.9495) / 20.0505
    float4 parameters = tex3D<float4>(prebaked_disk,
        phi_final * 0.15915494f,                     // x → phi
        fabsf(cam_pos.z) / 2.5f,                     // y → z
        (r_disk - 4.9495f) / 20.0505f);              // z → r_disk

    float4 emission = disk_emission(fmaxf(parameters.y*g, 1000.0f), parameters.z*g*g*g*g, lut_color);

    // ---- 体积渲染积分（与原代码一致） ----
    float ravg = (length(prev_pos)+r)/2.0f;
    float uuu = 1.0f + 1.0f/(2.0f*ravg);

    float k = 2.0f;
    float intensity_factor = 1.0f - __expf(-(k * parameters.z*g*g*g*g)*(k * parameters.z*g*g*g*g));
    float step_opacity = parameters.x * 1.7f*uuu*uuu*length(cam_pos-prev_pos)* intensity_factor / g;
    float temp_fade = __saturatef((parameters.y * g - 1400.0f) / 500.0f);
    step_opacity *= temp_fade;
    float alpha = 1.0f - __expf(-step_opacity);
    float transmittance = 1.0f - accumulated_color.w;
    accumulated_color.x += emission.x * alpha * transmittance;
    accumulated_color.y += emission.y * alpha * transmittance;
    accumulated_color.z += emission.z * alpha * transmittance;
    accumulated_color.w += alpha * transmittance;


    if (accumulated_color.w > 0.99f) {
        flag = false;
    }
}




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




color = accumulated_color + bkgd * (1.0f - accumulated_color.w);

} else {

color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
}

buffer = buffer+color;
}
buffer = buffer*(1.0f/(float)jitternum);
    int pixel_index = (pixel_idy * imgwidth + pixel_idx);
    raw_img[pixel_index] = make_float4(buffer.x,buffer.y,buffer.z,0.0);
}



extern "C"
__global__ void taaColorClampingKernel(
    const float4* currentFrame,
    const float4* prevFrame,
    float4* outputFrame,
    int width, int height,
    float alpha,
    int frames
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float3 min_color = make_float3(1e10f, 1e10f, 1e10f);
    float3 max_color = make_float3(-1e10f, -1e10f, -1e10f);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = fmaxf(0, fminf(x+dx, width-1));
            int ny = fmaxf(0, fminf(y+dy, height-1));

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
