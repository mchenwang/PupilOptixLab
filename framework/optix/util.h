#pragma once

#include <optix.h>
#include "cuda/preprocessor.h"
#include "cuda/vec_math.h"

namespace Pupil::optix {
constexpr float EPS = 0.000001f;
constexpr const float MAX_DISTANCE = 1e16f;

CUDA_INLINE CUDA_DEVICE void PackPointer(void *target, uint32_t &u0, uint32_t &u1) noexcept {
    const uint64_t ptr = reinterpret_cast<uint64_t>(target);
    u0 = ptr >> 32;
    u1 = ptr & 0x00000000ffffffff;
}

CUDA_INLINE CUDA_DEVICE void *UnpackPointer(uint32_t u0, uint32_t u1) noexcept {
    const uint64_t ptr = static_cast<uint64_t>(u0) << 32 | u1;
    return reinterpret_cast<void *>(ptr);
}

#ifdef PUPIL_OPTIX
template<typename T>
CUDA_INLINE CUDA_DEVICE T *GetPRD() noexcept {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(UnpackPointer(u0, u1));
}
#endif

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

CUDA_INLINE CUDA_HOSTDEVICE float3 Refract(float3 v, float cos_theta_t, float eta) noexcept {
    float scale = -(cos_theta_t < 0.f ? 1.f / eta : eta);
    return normalize(make_float3(scale * v.x, scale * v.y, cos_theta_t));
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Refract(float3 v, float3 normal, float cos_theta_t, float eta) {
    if (cos_theta_t < 0) eta = 1 / eta;
    return normal * (dot(v, normal) * eta + cos_theta_t) - v * eta;
}

// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
CUDA_INLINE CUDA_HOSTDEVICE void BuildONB(float3 N, float3 &b1, float3 &b2) noexcept {
    float sign = copysignf(1.f, N.z);
    float a = -1.f / (sign + N.z);
    float b = N.x * N.y * a;
    b1 = make_float3(1.f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    b2 = make_float3(b, sign + N.y * N.y * a, -N.y);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ToLocal(float3 v, float3 N) {
    float3 b1;
    float3 b2;
    BuildONB(N, b1, b2);
    return make_float3(dot(v, b1), dot(v, b2), dot(v, N));
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ToWorld(float3 v, float3 N) {
    float3 b1;
    float3 b2;
    BuildONB(N, b1, b2);
    return b1 * v.x + b2 * v.y + N * v.z;
}

CUDA_INLINE CUDA_HOSTDEVICE float2 GetSphereTexcoord(float3 local_p) noexcept {
    float phi = atan2(local_p.y, local_p.x);
    phi = phi < 0.f ? phi + M_PIf * 2.f : phi;
    auto t = local_p;
    t.z -= t.z >= 0.f ? 1.f : -1.f;
    // float theta = asin(0.5f * length(t)) * 2.f;
    // theta = local_p.z >= 0.f ? theta : M_PIf - theta;
    float theta = acos(local_p.z);

    // make_float2(asin(ret.normal.x) * M_1_PIf + 0.5f, asin(ret.normal.y) * M_1_PIf + 0.5f);
    return make_float2(phi * M_1_PIf * 0.5f, theta * M_1_PIf);
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

CUDA_INLINE CUDA_HOSTDEVICE float3 ACESToneMapping(float3 color, float adapted_lum) noexcept {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    color *= adapted_lum;
    return (color * (A * color + B)) / (color * (C * color + D) + E);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 GammaCorrection(float3 color, float gamma) {
    return make_float3(powf(color.x, 1.f / gamma), powf(color.y, 1.f / gamma), powf(color.z, 1.f / gamma));
}

CUDA_INLINE CUDA_HOSTDEVICE float GetLuminance(float3 color) noexcept {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

CUDA_INLINE CUDA_HOSTDEVICE float MISWeight(float x, float y) {
    return x / (x + y);
}

CUDA_INLINE CUDA_HOSTDEVICE bool IsZero(float v) noexcept {
    return abs(v) < EPS;
}

CUDA_INLINE CUDA_HOSTDEVICE bool IsZero(float2 v) noexcept {
    return abs(v.x) < EPS && abs(v.y) < EPS;
}

CUDA_INLINE CUDA_HOSTDEVICE bool IsZero(float3 v) noexcept {
    return abs(v.x) < EPS && abs(v.y) < EPS && abs(v.z) < EPS;
}

CUDA_INLINE CUDA_HOSTDEVICE float Lerp(const float &a, const float &b, const float t) noexcept {
    return a + t * (b - a);
}
}// namespace Pupil::optix