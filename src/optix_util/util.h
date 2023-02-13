#pragma once

#include <optix.h>
#include "cuda_util/preprocessor.h"
#include "cuda_util/vec_math.h"

namespace optix_util {
CUDA_DEVICE static const float EPS = 0.000001f;

CUDA_INLINE CUDA_DEVICE void PackPointer(void *target, uint32_t &u0, uint32_t &u1) noexcept {
    const uint64_t ptr = reinterpret_cast<uint64_t>(target);
    u0 = ptr >> 32;
    u1 = ptr & 0x00000000ffffffff;
}

CUDA_INLINE CUDA_DEVICE void *UnpackPointer(uint32_t u0, uint32_t u1) noexcept {
    const uint64_t ptr = static_cast<uint64_t>(u0) << 32 | u1;
    return reinterpret_cast<void *>(ptr);
}

#ifndef PUPIL_OPTIX_LAUNCHER_SIDE
template<typename T>
CUDA_INLINE CUDA_DEVICE T *GetPRD() noexcept {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(UnpackPointer(u0, u1));
}
#endif// PUPIL_OPTIX_LAUNCHER_SIDE

// https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
// return interpolation factor for triangle vertex
CUDA_INLINE CUDA_HOSTDEVICE float3 UniformSampleTriangle(const float u1, const float u2) noexcept {
    const float sqrt_u1 = sqrtf(u1);
    return make_float3(1.f - sqrt_u1, sqrt_u1 * (1.f - u2), u2 * sqrt_u1);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 UniformSampleSphere(const float u1, const float u2) noexcept {
    const float z = 1.f - 2.f * u1;
    const float sin_theta = sqrtf(max(0.f, 1.f - z * z));
    const float phi = 2.f * M_PIf * u2;
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Onb(const float3 normal, const float3 p) noexcept {
    float3 binormal;
    if (fabs(normal.x) > fabs(normal.z))
        binormal = make_float3(-normal.y, normal.x, 0.f);
    else
        binormal = make_float3(0.f, -normal.z, normal.y);

    binormal = normalize(binormal);
    float3 tangent = cross(binormal, normal);

    return p.x * tangent + p.y * binormal + p.z * normal;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 CosineSampleHemisphere(const float u1, const float u2) noexcept {
    float3 p{ 0.f };
    const float sin_theta = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = sin_theta * cosf(phi);
    p.y = sin_theta * sinf(phi);
    p.z = sqrtf(fmaxf(0.f, 1.f - sin_theta * sin_theta));

    return p;
}
CUDA_INLINE CUDA_HOSTDEVICE float CosineSampleHemispherePdf(float3 v) noexcept {
    return v.z > 0.f ? M_1_PIf * v.z : 0.f;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 UniformSampleHemisphere(const float u1, const float u2) noexcept {
    float3 p{ 0.f };
    const float z = 1.f - 2.f * u1;
    const float sin_theta = sqrtf(fmaxf(0.f, 1.f - z * z));
    const float phi = 2.0f * M_PIf * u2;
    p.x = sin_theta * cosf(phi);
    p.y = sin_theta * sinf(phi);
    p.z = abs(z);

    return p;
}
CUDA_INLINE CUDA_HOSTDEVICE float UniformSampleHemispherePdf(float3 v) noexcept {
    return v.z > 0.f ? M_1_PIf * 0.5f : 0.f;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Reflect(float3 v) noexcept {
    v.x = -v.x;
    v.y = -v.y;
    return v;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Reflect(float3 v, float3 normal) noexcept {
    return -v + 2 * dot(v, normal) * normal;
}

CUDA_INLINE CUDA_HOSTDEVICE void CoordinateSystem(const float3 &a, float3 &b, float3 &c) {
    if (abs(a.x) > abs(a.y)) {
        float invLen = 1.0f / sqrtf(a.x * a.x + a.z * a.z);
        c = make_float3(a.z * invLen, 0.0f, -a.x * invLen);
    } else {
        float inv_len = 1.0f / sqrtf(a.y * a.y + a.z * a.z);
        c = make_float3(0.0f, a.z * inv_len, -a.y * inv_len);
    }
    float3 _a = make_float3(a.x, a.y, a.z);
    c = normalize(c);
    b = cross(_a, c);
    b = normalize(b);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ToLocal(float3 v, float3 N) {
    float3 S;
    float3 T;
    CoordinateSystem(N, S, T);
    return make_float3(dot(v, S), dot(v, T), dot(v, N));
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ToWorld(float3 p, float3 N) {
    float3 S;
    float3 T;
    CoordinateSystem(N, S, T);
    return S * p.x + T * p.y + N * p.z;
}

// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
CUDA_INLINE CUDA_HOSTDEVICE float3 GetBarycentricCoordinates(float3 P, float3 A, float3 B, float3 C) noexcept {
    float3 v0 = B - A, v1 = C - A, v2 = P - A;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float3 bary;
    bary.y = (d11 * d20 - d01 * d21) / denom;
    bary.z = (d00 * d21 - d01 * d20) / denom;
    bary.x = 1.f - bary.y - bary.z;
    return bary;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ACESToneMapping(float3 color, float adaptedLum) noexcept {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    color *= adaptedLum;
    return (color * (A * color + B)) / (color * (C * color + D) + E);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 GammaCorrection(float3 color, float gamma) {
    return make_float3(powf(color.x, 1.f / gamma), powf(color.y, 1.f / gamma), powf(color.z, 1.f / gamma));
}

CUDA_INLINE CUDA_HOSTDEVICE float MISWeight(float x, float y) {
    return x / (x + y);
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float v) noexcept {
    return abs(v) < EPS;
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float2 v) noexcept {
    return abs(v.x) < EPS && abs(v.y) < EPS;
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float3 v) noexcept {
    return abs(v.x) < EPS && abs(v.y) < EPS && abs(v.z) < EPS;
}
}// namespace optix_util