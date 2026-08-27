#pragma once

// #include <cuda_runtime.h>
// #include <math_functions.h>

static __device__ __forceinline__ float3 normalize(float3 v)
{
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

static __device__ __forceinline__ float length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static __device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __device__ __forceinline__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static __device__ __forceinline__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __device__ __forceinline__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __device__ __forceinline__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

static __device__ __forceinline__ float3 operator/(float3 a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

static __device__ __forceinline__ float4 operator/(float4 a, float s)
{
    return make_float4(a.x / s, a.y / s, a.z / s, a.w / s);
}

static __device__ __forceinline__ float operator*(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 fma_3s3(float3 a, float s, float3 b)
{
    return make_float3(__fmaf_rn(a.x, s, b.x), __fmaf_rn(a.y, s, b.y), __fmaf_rn(a.z, s, b.z));
}

__device__ __forceinline__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}