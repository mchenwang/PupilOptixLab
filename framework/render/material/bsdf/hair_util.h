#pragma once

#include "../predefine.h"
#include "../fresnel.h"
#include "optix/util.h"

namespace Pupil::optix::material {
CUDA_INLINE CUDA_HOSTDEVICE float I0(float x) {
    float val = 0.f;
    float x2i = 1.f;
    auto ifact = 1ull;
    auto i4 = 1u;
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= i;
        val += x2i / (i4 * Sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

CUDA_INLINE CUDA_HOSTDEVICE float LogI0(float x) {
    if (x > 12.f)
        return x + 0.5f * (-log(2.f * M_PIf) + log(1.f / x) + 1.f / (8.f * x));
    else
        return log(I0(x));
}

CUDA_INLINE CUDA_HOSTDEVICE float Mp(float cos_theta_i, float cos_theta_o, float sin_theta_i, float sin_theta_o, float v) {
    float a = cos_theta_i * cos_theta_o / v;
    float b = sin_theta_i * sin_theta_o / v;
    float mp = 0.f;
    if (v <= 0.1f)
        mp = exp(LogI0(a) - b - 1.f / v + 0.6931f + log(1.f / (2.f * v)));
    else
        mp = (exp(-b) * I0(a)) / (sinh(1.f / v) * 2.f * v);

    return mp;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Ap0(float fresnel) {
    return make_float3(fresnel, fresnel, fresnel);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Ap1(float fresnel, float3 &T) {
    return Sqr(1.f - fresnel) * T;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 Ap2(float fresnel, float3 &T) {
    return Sqr(1.f - fresnel) * T * T * fresnel;
}

CUDA_INLINE CUDA_HOSTDEVICE float3 ApMax(float fresnel, float3 &T) {
    float3 ap2 = Ap2(fresnel, T);
    return ap2 * fresnel * T / (make_float3(1.f, 1.f, 1.f) - T * fresnel);
}

CUDA_INLINE CUDA_HOSTDEVICE float Phi(int p, float gamma_o, float gamma_t) {
    return 2.f * p * gamma_t - 2.f * gamma_o + p * M_PIf;
}

CUDA_INLINE CUDA_HOSTDEVICE float Logistic(float x, float s) {
    x = abs(x);
    return exp(-x / s) / (s * Sqr(1.f + exp(-x / s)));
}

CUDA_INLINE CUDA_HOSTDEVICE float LogisticCDF(float x, float s) {
    return 1.f / (1.f + exp(-x / s));
}

CUDA_INLINE CUDA_HOSTDEVICE float TrimmedLogistic(float x, float s, float a, float b) {
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

CUDA_INLINE CUDA_HOSTDEVICE float Np(float phi, int p, float s, float gamma_o, float gamma_t) {
    float dphi = phi - Phi(p, gamma_o, gamma_t);
    while (dphi > M_PIf) dphi -= 2.f * M_PIf;
    while (dphi < -M_PIf) dphi += 2.f * M_PIf;
    return TrimmedLogistic(dphi, s, -M_PIf, M_PIf);
}

CUDA_INLINE CUDA_HOSTDEVICE float4 ComputeApPdf(float3 sigma_a, float sin_theta_o, float cos_theta_o, float cos_gamma_o, float eta, float h) {
    float sin_theta_t = sin_theta_o / eta;
    float cos_theta_t = SafeSqrt(1.f - Sqr(sin_theta_t));

    float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
    float sin_gamma_t = h / etap;
    float cos_gamma_t = SafeSqrt(1.f - Sqr(sin_gamma_t));

    float fac = (2.f * cos_gamma_t / cos_theta_t);
    float3 T = optix::Exp(-sigma_a * fac);

    float fresnel = fresnel::DielectricReflectance(eta, cos_theta_o * cos_gamma_o);

    float3 ap_vec[4];
    ap_vec[0] = Ap0(fresnel);
    ap_vec[1] = Ap1(fresnel, T);
    ap_vec[2] = Ap2(fresnel, T);
    ap_vec[3] = ApMax(fresnel, T);

    float4 ap;
    ap.x = (ap_vec[0].x + ap_vec[0].y + ap_vec[0].z) / 3.f;
    ap.y = (ap_vec[1].x + ap_vec[1].y + ap_vec[1].z) / 3.f;
    ap.z = (ap_vec[2].x + ap_vec[2].y + ap_vec[2].z) / 3.f;
    ap.w = (ap_vec[3].x + ap_vec[3].y + ap_vec[3].z) / 3.f;

    float sum = ap.x + ap.y + ap.z + ap.w;
    ap /= sum;

    return ap;
}

CUDA_INLINE CUDA_HOSTDEVICE float SampleTrimmedLogistic(float u, float s, float a, float b) {
    float k = LogisticCDF(b, s) - LogisticCDF(a, s);
    float x = -s * log(1.f / (u * k + LogisticCDF(a, s)) - 1.f);

    return clamp(x, a, b);
}
}// namespace Pupil::optix::material