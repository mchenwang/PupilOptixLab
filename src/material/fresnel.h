#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/vec_math.h"

namespace optix_util::material::fresnel {
CUDA_INLINE CUDA_HOSTDEVICE float DielectricReflectance(float eta, float cos_theta_i, float &cos_theta_t) {
    // if (cos_theta_i < 0.f) {
    //     eta = 1.f / eta;
    //     cos_theta_i = -cos_theta_i;
    // }
    // float sinThetaTSq = eta * eta * (1.f - cos_theta_i * cos_theta_i);
    // if (sinThetaTSq > 1.f) {
    //     cos_theta_t = 0.f;
    //     return 1.f;
    // }
    // cos_theta_t = sqrtf(max(1.f - sinThetaTSq, 0.f));

    // float Rs = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    // float Rp = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);

    // return (Rs * Rs + Rp * Rp) * 0.5f;
    float scale = cos_theta_i > 0.f ? 1.f / eta : eta;
    float cos_theta_t2 = 1.f - (1.f - cos_theta_i * cos_theta_i) * (scale * scale);

    float o_cos_theta_i = cos_theta_i;
    cos_theta_i = abs(cos_theta_i);
    cos_theta_t = sqrtf(fmaxf(0.f, cos_theta_t2));

    float rs = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    cos_theta_t = o_cos_theta_i > 0.f ? -cos_theta_t : cos_theta_t;
    return 0.5f * (rs * rs + rp * rp);
}
CUDA_INLINE CUDA_HOSTDEVICE float DielectricReflectance(float eta, float cos_theta_i) {
    float cos_theta_t;
    return DielectricReflectance(eta, cos_theta_i, cos_theta_t);
}

CUDA_INLINE CUDA_HOSTDEVICE float ConductorReflectance(float eta, float k, float cos_theta_i) {
    // float cos_theta_i_sq = cos_theta_i * cos_theta_i;
    // float sin_theta_i_sq = max(1.f - cos_theta_i_sq, 0.f);
    // float sin_theta_i_qu = sin_theta_i_sq * sin_theta_i_sq;

    // float inner = eta * eta - k * k - sin_theta_i_sq;
    // float a_sq_plus_b_sq = sqrtf(max(inner * inner + 4.f * eta * eta * k * k, 0.f));
    // float a = sqrtf(max((a_sq_plus_b_sq + inner) * 0.5f, 0.f));

    // float Rs = ((a_sq_plus_b_sq + cos_theta_i_sq) - (2.f * a * cos_theta_i)) /
    //            ((a_sq_plus_b_sq + cos_theta_i_sq) + (2.f * a * cos_theta_i));
    // float Rp = ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) - (2.f * a * cos_theta_i * sin_theta_i_sq)) /
    //            ((cos_theta_i_sq * a_sq_plus_b_sq + sin_theta_i_qu) + (2.f * a * cos_theta_i * sin_theta_i_sq));

    // return 0.5f * (Rs + Rs * Rp);
    float cos_theta_i2 = cos_theta_i * cos_theta_i;
    float sin_theta_i2 = 1.f - cos_theta_i2;
    float sin_theta_i4 = sin_theta_i2 * sin_theta_i2;

    float t1 = eta * eta - k * k - sin_theta_i2;
    float a2pb2 = sqrtf(fmaxf(0.f, t1 * t1 + 4.f * k * k * eta * eta));
    float a = sqrtf(fmaxf(0.f, 0.5f * (a2pb2 + t1)));

    float term1 = a2pb2 + cos_theta_i2;
    float term2 = 2.f * a * cos_theta_i;
    float rs2 = (term1 - term2) / (term1 + term2);

    float term3 = a2pb2 * cos_theta_i2 + sin_theta_i4;
    float term4 = term2 * sin_theta_i2;
    float rp2 = rs2 * (term3 - term4) / (term3 + term4);

    return 0.5f * (rp2 + rs2);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ConductorReflectance(const float3 eta, const float3 k, float cos_theta_i) {
    return make_float3(
        ConductorReflectance(eta.x, k.x, cos_theta_i),
        ConductorReflectance(eta.y, k.y, cos_theta_i),
        ConductorReflectance(eta.z, k.z, cos_theta_i));
}

CUDA_INLINE CUDA_HOSTDEVICE float DiffuseReflectance(float eta) {
    if (eta < 1) {
        /* Fit by Egan and Hilgeman (1973). Works
               reasonably well for "normal" IOR values (<2).

               Max rel. error in 1.0 - 1.5 : 0.1%
               Max rel. error in 1.5 - 2   : 0.6%
               Max rel. error in 2.0 - 5   : 9.5%
            */
        return -1.4399f * (eta * eta) + 0.7099f * eta + 0.6681f + 0.0636f / eta;
    } else {
        /* Fit by d'Eon and Irving (2011)
             *
             * Maintains a good accuracy even for
             * unrealistic IOR values.
             *
             * Max rel. error in 1.0 - 2.0   : 0.1%
             * Max rel. error in 2.0 - 10.0  : 0.2%
             */
        float invEta = 1.0f / eta;
        float invEta2 = invEta * invEta;
        float invEta3 = invEta2 * invEta;
        float invEta4 = invEta3 * invEta;
        float invEta5 = invEta4 * invEta;

        return 0.919317f - 3.4793f * invEta + 6.75335f * invEta2 - 7.80989f * invEta3 + 4.98554f * invEta4 - 1.36881f * invEta5;
    }
}
}// namespace optix_util::material::fresnel