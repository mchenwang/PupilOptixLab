#pragma once

#include "cuda/preprocessor.h"
#include "cuda/vec_math.h"

namespace Pupil::optix::material::fresnel {
CUDA_INLINE CUDA_HOSTDEVICE float DielectricReflectance(float eta, float cos_theta_i, float &cos_theta_t) {
    float scale = cos_theta_i > 0.f ? 1.f / eta : eta;
    float cos_theta_t2 = 1.f - (1.f - cos_theta_i * cos_theta_i) * (scale * scale);

    if (cos_theta_t2 <= 0.0f) {
        cos_theta_t = 0.0f;
        return 1.0f;
    }

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
        float inv_eta = 1.0f / eta;
        float inv_eta2 = inv_eta * inv_eta;
        float inv_eta3 = inv_eta2 * inv_eta;
        float inv_eta4 = inv_eta3 * inv_eta;
        float inv_eta5 = inv_eta4 * inv_eta;

        return 0.919317f - 3.4793f * inv_eta + 6.75335f * inv_eta2 - 7.80989f * inv_eta3 + 4.98554f * inv_eta4 - 1.36881f * inv_eta5;
    }
}
}// namespace Pupil::optix::material::fresnel