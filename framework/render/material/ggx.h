#pragma once

#include "cuda/preprocessor.h"
#include "cuda/vec_math.h"

#define GGX_Sample_Visible_Area

namespace Pupil::optix::material::ggx {

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

CUDA_INLINE CUDA_HOSTDEVICE float Pdf(float3 wo, float3 wh, float alpha) noexcept {
#ifdef GGX_Sample_Visible_Area
    return D(wh, alpha) * G1(wo, alpha) * dot(wo, wh) / abs(wo.z);
#else
    return D(wh, alpha) * abs(wh.z);
#endif
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Sample(float3 wo, float alpha, float2 xi) noexcept {
#ifdef GGX_Sample_Visible_Area
    // sample GGX VNDF
    float3 vh = normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));

    float3 T1 = wo.z < 0.9999f ? normalize(cross(make_float3(0.f, 0.f, 1.f), vh)) : make_float3(1.f, 0.f, 0.f);
    float3 T2 = cross(vh, T1);

    float r = sqrtf(xi.x);
    float phi = 2.f * M_PIf * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    float3 nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * vh;
    float3 ne = make_float3(alpha * nh.x, alpha * nh.y, fmaxf(0.f, nh.z));
    return normalize(ne);
#else
    float phi = 2.f * M_PIf * xi.y;
    float tan_theta2 = alpha * alpha * xi.x / (1.f - xi.x);
    float cos_theta = 1.f / sqrtf(1.f + tan_theta2);
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
#endif
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

CUDA_INLINE CUDA_HOSTDEVICE float Pdf(float3 wo, float3 wh, float2 alpha) noexcept {
#ifdef GGX_Sample_Visible_Area
    return D(wh, alpha) * G1(wo, alpha) * dot(wo, wh) / abs(wo.z);
#else
    return D(wh, alpha) * abs(wh.z);
#endif
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Sample(float3 wo, float2 alpha, float2 xi) noexcept {
#ifdef GGX_Sample_Visible_Area
    // sample GGX VNDF
    float3 vh = normalize(make_float3(alpha.x * wo.x, alpha.y * wo.y, wo.z));

    float3 T1 = wo.z < 0.9999f ? normalize(cross(make_float3(0.f, 0.f, 1.f), vh)) : make_float3(1.f, 0.f, 0.f);
    float3 T2 = cross(vh, T1);

    float r = sqrtf(xi.x);
    float phi = 2.f * M_PIf * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    float3 nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * vh;
    float3 ne = make_float3(alpha.x * nh.x, alpha.y * nh.y, fmaxf(0.f, nh.z));
    return normalize(ne);
#else
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
#endif
}

}// namespace Pupil::optix::material::ggx