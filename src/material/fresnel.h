#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/vec_math.h"

namespace optix_util::material::fresnel {
CUDA_INLINE CUDA_HOSTDEVICE float DielectricReflectance(float eta, float cos_theta_i, float &cos_theta_t) {
    if (cos_theta_i < 0.f) {
        eta = 1.f / eta;
        cos_theta_i = -cos_theta_i;
    }
    float sinThetaTSq = eta * eta * (1.f - cos_theta_i * cos_theta_i);
    if (sinThetaTSq > 1.f) {
        cos_theta_t = 0.f;
        return 1.f;
    }
    cos_theta_t = sqrtf(max(1.f - sinThetaTSq, 0.f));

    float Rs = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float Rp = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);

    return (Rs * Rs + Rp * Rp) * 0.5f;
}
CUDA_INLINE CUDA_HOSTDEVICE float DielectricReflectance(float eta, float cos_theta_i) {
    float cos_theta_t;
    return DielectricReflectance(eta, cos_theta_i, cos_theta_t);
}

CUDA_INLINE CUDA_HOSTDEVICE float ConductorReflectance(float eta, float k, float cos_theta_i) {
    float cos_theta_i_sq = cos_theta_i * cos_theta_i;
    float sin_theta_i_sq = max(1.f - cos_theta_i_sq, 0.f);
    float sin_theta_i_qu = sin_theta_i_sq * sin_theta_i_sq;

    float inner = eta * eta - k * k - sin_theta_i_sq;
    float a_sq_plus_b_sq = sqrtf(max(inner * inner + 4.f * eta * eta * k * k, 0.f));
    float a = sqrtf(max((a_sq_plus_b_sq + inner) * 0.5f, 0.f));

    float Rs = ((a_sq_plus_b_sq + cos_theta_i_sq) - (2.f * a * cos_theta_i)) /
               ((a_sq_plus_b_sq + cos_theta_i_sq) + (2.f * a * cos_theta_i));
    float Rp = ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) - (2.f * a * cos_theta_i * sin_theta_i_sq)) /
               ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) + (2.f * a * cos_theta_i * sin_theta_i_sq));

    return 0.5f * (Rs + Rs * Rp);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ConductorReflectance(const float3 eta, const float3 k, float cos_theta_i) {
    return make_float3(
        ConductorReflectance(eta.x, k.x, cos_theta_i),
        ConductorReflectance(eta.y, k.y, cos_theta_i),
        ConductorReflectance(eta.z, k.z, cos_theta_i));
}
}// namespace optix_util::material::fresnel