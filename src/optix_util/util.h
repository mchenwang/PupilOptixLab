#pragma once

#include <optix.h>

#include "cuda_util/vec_math.h"

namespace optix_util {
__device__ static const float EPS = 0.000001f;

__forceinline__ __device__ void PackPointer(void *target, uint32_t &u0, uint32_t &u1) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(target);
    u0 = ptr >> 32;
    u1 = ptr & 0x00000000ffffffff;
}

__forceinline__ __device__ void *UnpackPointer(uint32_t u0, uint32_t u1) {
    const uint64_t ptr = static_cast<uint64_t>(u0) << 32 | u1;
    return reinterpret_cast<void *>(ptr);
}

template<typename T>
__forceinline__ __device__ T *GetPRD() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(UnpackPointer(u0, u1));
}

__forceinline__ __device__ float3 Onb(const float3 normal, const float3 p) {
    float3 binormal;
    if (fabs(normal.x) > fabs(normal.z))
        binormal = make_float3(-normal.y, normal.x, 0.f);
    else
        binormal = make_float3(0.f, -normal.z, normal.y);

    binormal = normalize(binormal);
    float3 tangent = cross(binormal, normal);

    return p.x * tangent + p.y * binormal + p.z * normal;
}

__forceinline__ __device__ float3 CosineSampleHemisphere(const float u1, const float u2, float3 normal) {
    float3 p{ 0.f };
    const float sin_theta = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = sin_theta * cosf(phi);
    p.y = sin_theta * sinf(phi);
    p.z = sqrtf(fmaxf(0.f, 1.f - sin_theta * sin_theta));

    return Onb(normal, p);
}

__forceinline__ __device__ float3 CosineSampleHemisphere(const float u1, const float u2) {
    float3 p{ 0.f };
    const float sin_theta = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = sin_theta * cosf(phi);
    p.y = sin_theta * sinf(phi);
    p.z = sqrtf(fmaxf(0.f, 1.f - sin_theta * sin_theta));

    return p;
}

__forceinline__ __device__ float3 SampleUniformSphere(const float u1, const float u2) {
    float z = 1.f - 2.f * u1;
    float r = sqrtf(max(0.f, 1.f - z * z));
    float phi = 2 * M_PIf * u2;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

__forceinline__ __device__ float MISWeight(float x, float y) {
    return x / (x + y);
}

__forceinline__ __device__ float Saturate(float x) {
    return fmaxf(0.f, fminf(1.f, x));
}

__device__ inline void CoordinateSystem(const float3 &a, float3 &b, float3 &c) {
    if (abs(a.x) > abs(a.y)) {
        float invLen = 1.0f / sqrtf(a.x * a.x + a.z * a.z);
        c = make_float3(a.z * invLen, 0.0f, -a.x * invLen);
    } else {
        float invLen = 1.0f / sqrtf(a.y * a.y + a.z * a.z);
        c = make_float3(0.0f, a.z * invLen, -a.y * invLen);
    }
    float3 _a = make_float3(a.x, a.y, a.z);
    c = normalize(c);
    b = cross(_a, c);
    b = normalize(b);
}

__device__ inline float3 ToLocal(float3 v, float3 N) {
    float3 S;
    float3 T;
    CoordinateSystem(N, S, T);
    return make_float3(dot(v, S), dot(v, T), dot(v, N));
}

__device__ inline float3 ToWorld(float3 p, float3 N) {
    // float3 up = abs(N.z) < 0.999 ? make_float3(0.0, 0.0, 1.0) : make_float3(1.0, 0.0, 0.0);
    // float3 tangentX = normalize(cross(up, N));
    // float3 tangentY = cross(N, tangentX);

    // float3 sampleVec = tangentX * p.x + tangentY * p.y + N * p.z;
    // sampleVec = normalize(sampleVec);
    float3 S;
    float3 T;
    CoordinateSystem(N, S, T);
    float3 sampleVec = S * p.x + T * p.y + N * p.z;
    return sampleVec;
}

__forceinline__ __device__ bool IsZero(float v) {
    return abs(v) < EPS;
}

__forceinline__ __device__ bool IsZero(float2 v) {
    return abs(v.x) < EPS && abs(v.y) < EPS;
}

__forceinline__ __device__ bool IsZero(float3 v) {
    return abs(v.x) < EPS && abs(v.y) < EPS && abs(v.z) < EPS;
}
}// namespace optix_util