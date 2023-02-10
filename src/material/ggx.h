#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/vec_math.h"

namespace optix_util::material::ggx {

CUDA_INLINE CUDA_HOSTDEVICE float Lambda(float3 w, float alpha) noexcept {
    float a2 = alpha * alpha;
    float3 v2 = w * w;
    return (-1.f + sqrtf(1.f + (v2.x + v2.y) * a2 / v2.z)) / 2.f;
}

CUDA_INLINE CUDA_HOSTDEVICE float G1(float3 w, float alpha) noexcept {
    return 1.f / (1.f + Lambda(w, alpha));
}

CUDA_INLINE CUDA_HOSTDEVICE float G(float3 wi, float3 wo, float alpha) noexcept {
    return G1(wi, alpha) * G1(wo, alpha);
}

CUDA_INLINE CUDA_HOSTDEVICE float D(float3 wh, float alpha) noexcept {
    float a2 = alpha * alpha;
    float3 v2 = wh * wh;
    float t = (v2.x + v2.y) / a2 + v2.z;
    return 1.f / (M_PIf * a2 * t * t);
}

CUDA_INLINE CUDA_HOSTDEVICE float Pdf(float3 wh, float alpha) noexcept {
    return D(wh, alpha) * abs(wh.z);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Sample(float3 wo, float alpha, float2 xi) noexcept {
    float phi = 2.f * M_PIf * xi.y;
    float tan_theta2 = alpha * alpha * xi.x / (1.f - xi.x);
    float cos_theta = 1.f / sqrtf(1.f + tan_theta2);
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

CUDA_INLINE CUDA_HOSTDEVICE float Lambda(float3 w, float2 alpha) noexcept {
    float ax2 = alpha.x * alpha.x;
    float ay2 = alpha.y * alpha.y;
    float3 v2 = w * w;
    return (-1.f + sqrtf(1.f + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2.f;
}

CUDA_INLINE CUDA_HOSTDEVICE float G1(float3 w, float2 alpha) noexcept {
    return 1.f / (1.f + Lambda(w, alpha));
}

CUDA_INLINE CUDA_HOSTDEVICE float G(float3 wi, float3 wo, float2 alpha) noexcept {
    return G1(wi, alpha) * G1(wo, alpha);
}

CUDA_INLINE CUDA_HOSTDEVICE float D(float3 wh, float2 alpha) noexcept {
    float ax2 = alpha.x * alpha.x;
    float ay2 = alpha.y * alpha.y;
    float3 v2 = wh * wh;
    return 1.f / (M_PIf * alpha.x * alpha.y * (v2.x / ax2 + v2.y / ay2 + v2.z) * (v2.x / ax2 + v2.y / ay2 + v2.z));
}

CUDA_INLINE CUDA_HOSTDEVICE float Pdf(float3 wh, float2 alpha) noexcept {
    return D(wh, alpha) * abs(wh.z);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Sample(float3 wo, float2 alpha, float2 xi) noexcept {
    float phi = atan(alpha.y / alpha.x * tan(2 * M_PIf * xi.y + 0.5f * M_PIf));
    if (xi.y > 0.5f) phi += M_PIf;
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    float x2 = alpha.x * alpha.x;
    float y2 = alpha.y * alpha.y;
    float a2 = 1.f / (cos_phi * cos_phi / x2 + sin_phi * sin_phi / y2);
    float tan_theta2 = a2 * xi.x / (1.f - xi.x);
    float cos_theta = 1.f / sqrtf(1.f + tan_theta2);
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    return make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

}// namespace optix_util::material::ggx