#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include <cuda_runtime.h>

namespace Pupil::optix::material {
struct HairAttr {
    cuda::Texture sigma_a;

    float3 longitudinal_v;
    float3 sin_2k_alpha;
    float3 cos_2k_alpha;

    float azimuthal_s;

    struct Local {
        float3 sigma_a;
        float3 longitudinal_v;
        float3 sin_2k_alpha;
        float3 cos_2k_alpha;

        float eta;
        float h;
        float azimuthal_s;

        static CUDA_INLINE CUDA_HOSTDEVICE float Fresnel(float eta, float cosTheta) {
            float F0 = Sqr(1 - eta) / Sqr(1 + eta);
            float xd = (1 - cosTheta);
            return F0 + (1 - F0) * xd * xd * xd * xd * xd;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float I0(float x) {
            float val = 0;
            float x2i = 1;
            int64_t ifact = 1;
            int i4 = 1;
            for (int i = 0; i < 10; ++i) {
                if (i > 1) ifact *= i;
                val += x2i / (i4 * Sqr(ifact));
                x2i *= x * x;
                i4 *= 4;
            }
            return val;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float LogI0(float x) {
            if (x > 12)
                return x + 0.5 * (-log(2 * M_PIf) + log(1 / x) + 1 / (8 * x));
            else
                return log(I0(x));
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float Mp(float cos_theta_i, float cos_theta_o, float sin_theta_i, float sin_theta_o, float v) {
            float a = cos_theta_i * cos_theta_o / v;
            float b = sin_theta_i * sin_theta_o / v;
            float mp = 0.f;
            if (v <= 0.1f)
                mp = exp(LogI0(a) - b - 1 / v + 0.6931f + log(1 / (2 * v)));
            else
                mp = (exp(-b) * I0(a)) / (sinh(1 / v) * 2 * v);

            return mp;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE
            float3
            Ap0(float fresnel, float3 &T) {
            return make_float3(fresnel, fresnel, fresnel);
        }

        static CUDA_INLINE CUDA_HOSTDEVICE
            float3
            Ap1(float fresnel, float3 &T) {
            return Sqr(1 - fresnel) * T;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE
            float3
            Ap2(float fresnel, float3 &T) {
            return Sqr(1 - fresnel) * T * T * fresnel;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE
            float3
            ApMax(float fresnel, float3 &T) {
            float3 ap2 = Ap2(fresnel, T);
            return ap2 * fresnel * T / (float3(1.f, 1.f, 1.f) - T * fresnel);
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float Phi(int p, float gamma_o, float gamma_t) {
            return 2 * p * gamma_t - 2 * gamma_o + p * M_PIf;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float Logistic(float x, float s) {
            x = abs(x);
            return exp(-x / s) / (s * Sqr(1 + exp(-x / s)));
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float LogisticCDF(float x, float s) {
            return 1 / (1 + exp(-x / s));
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float TrimmedLogistic(float x, float s, float a, float b) {
            return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float Np(float phi, int p, float s, float gamma_o, float gamma_t) {
            float dphi = phi - Phi(p, gamma_o, gamma_t);
            while (dphi > M_PIf) dphi -= 2 * M_PIf;
            while (dphi < -M_PIf) dphi += 2 * M_PIf;
            return TrimmedLogistic(dphi, s, -M_PIf, M_PIf);
        }

        static CUDA_INLINE CUDA_HOSTDEVICE
            float4
            computeApPdf(float3 sigma_a, float sin_theta_o, float cos_theta_o, float cos_gamma_o, float eta, float h) {
            float sin_theta_t = sin_theta_o / eta;
            float cosThetaT = SafeSqrt(1 - Sqr(sin_theta_t));

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float cos_gamma_t = SafeSqrt(1 - Sqr(sin_gamma_t));

            float fac = (2.f * cos_gamma_t / cosThetaT);
            float Tx = exp(-sigma_a.x * fac);
            float Ty = exp(-sigma_a.y * fac);
            float Tz = exp(-sigma_a.z * fac);
            float3 T = float3(Tx, Ty, Tz);

            float fresnel = Fresnel(eta, cos_theta_o * cos_gamma_o);

            float3 apVec[4] = { float3(0.f) };
            apVec[0] = Ap0(fresnel, T);
            apVec[1] = Ap1(fresnel, T);
            apVec[2] = Ap2(fresnel, T);
            apVec[3] = ApMax(fresnel, T);

            float4 ap;
            ap.x = (apVec[0].x + apVec[0].y + apVec[0].z) / 3.f;
            ap.y = (apVec[1].x + apVec[1].y + apVec[1].z) / 3.f;
            ap.z = (apVec[2].x + apVec[2].y + apVec[2].z) / 3.f;
            ap.w = (apVec[3].x + apVec[3].y + apVec[3].z) / 3.f;

            float sum = ap.x + ap.y + ap.z + ap.w;

            ap.x = ap.x / sum;
            ap.y = ap.y / sum;
            ap.z = ap.z / sum;
            ap.w = ap.w / sum;

            return ap;
        }

        static CUDA_INLINE CUDA_HOSTDEVICE float SampleTrimmedLogistic(float u, float s, float a, float b) {
            float k = LogisticCDF(b, s) - LogisticCDF(a, s);
            float x = -s * log(1 / (u * k + LogisticCDF(a, s)) - 1);

            return clamp(x, a, b);
        }

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;
            float3 wi = record.wi;
            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1 - Sqr(sin_theta_o));
            float phiO = atan2(wo.z, wo.y);

            float sin_theta_i = wi.x;
            float cos_theta_i = SafeSqrt(1 - Sqr(sin_theta_i));
            float phiI = atan2(wi.z, wi.y);

            float sin_theta_t = sin_theta_o / eta;
            float cosThetaT = SafeSqrt(1 - Sqr(sin_theta_t));

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float cos_gamma_t = SafeSqrt(1 - Sqr(sin_gamma_t));
            float gamma_t = SafeASin(sin_gamma_t);

            float fac = (2.f * cos_gamma_t / cosThetaT);
            float Tx = exp(-sigma_a.x * fac);
            float Ty = exp(-sigma_a.y * fac);
            float Tz = exp(-sigma_a.z * fac);
            float3 T = float3(Tx, Ty, Tz);

            float fresnel = fresnel::DielectricReflectance(eta, cos_theta_o * cos_gamma_o);

            float phi = phiI - phiO;
            float3 f(0.f);

            float sin_theta_op = 0.f, cos_theta_op = 0.f;

            sin_theta_op = sin_theta_o * cos_2k_alpha.y - cos_theta_o * sin_2k_alpha.y;
            cos_theta_op = cos_theta_o * cos_2k_alpha.y + sin_theta_o * sin_2k_alpha.y;
            cos_theta_op = abs(cos_theta_op);

            float Mp0 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.x);
            float Np0 = Np(phi, 0, azimuthal_s, gamma_o, gamma_t);
            float3 r_result = Mp0 * Ap0(fresnel, T) * Np0;
            f += r_result;

            sin_theta_op = sin_theta_o * cos_2k_alpha.x + cos_theta_o * sin_2k_alpha.x;
            cos_theta_op = cos_theta_o * cos_2k_alpha.x - sin_theta_o * sin_2k_alpha.x;
            cos_theta_op = abs(cos_theta_op);

            float Mp1 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.y);
            float Np1 = Np(phi, 1, azimuthal_s, gamma_o, gamma_t);
            float3 tt_result = Mp1 * Ap1(fresnel, T) * Np1;
            f += tt_result;

            sin_theta_op = sin_theta_o * cos_2k_alpha.z + cos_theta_o * sin_2k_alpha.z;
            cos_theta_op = cos_theta_o * cos_2k_alpha.z - sin_theta_o * sin_2k_alpha.z;
            cos_theta_op = abs(cos_theta_op);

            float Mp2 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.z);
            float Np2 = Np(phi, 2, azimuthal_s, gamma_o, gamma_t);
            float3 trt_result = Mp2 * Ap2(fresnel, T) * Np2;
            f += trt_result;

            float MpMax = Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, longitudinal_v.z);
            float NpMax = 1.f / (2.f * M_PIf);
            float3 max_result = MpMax * ApMax(fresnel, T) * NpMax;
            f += max_result;

            if (wi.z != 0)
                f /= abs(wi.z);

            record.f = f;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;
            float3 wi = record.wi;
            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1 - Sqr(sin_theta_o));
            float phiO = atan2(wo.z, wo.y);

            float sin_theta_i = wi.x;
            float cos_theta_i = SafeSqrt(1 - Sqr(sin_theta_i));
            float phiI = atan2(wi.z, wi.y);

            float sin_theta_t = sin_theta_o / eta;
            float cosThetaT = SafeSqrt(1 - Sqr(sin_theta_t));

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float cos_gamma_t = SafeSqrt(1 - Sqr(sin_gamma_t));
            float gamma_t = SafeASin(sin_gamma_t);

            float fac = (2.f * cos_gamma_t / cosThetaT);
            float Tx = exp(-sigma_a.x * fac);
            float Ty = exp(-sigma_a.y * fac);
            float Tz = exp(-sigma_a.z * fac);
            float3 T = float3(Tx, Ty, Tz);

            float fresnel = fresnel::DielectricReflectance(eta, cos_theta_o * cos_gamma_o);

            float phi = phiI - phiO;

            float sin_theta_op = 0.f, cos_theta_op = 0.f;

            sin_theta_op = sin_theta_o * cos_2k_alpha.y - cos_theta_o * sin_2k_alpha.y;
            cos_theta_op = cos_theta_o * cos_2k_alpha.y + sin_theta_o * sin_2k_alpha.y;
            cos_theta_op = abs(cos_theta_op);

            float Mp0 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.x);
            float Np0 = Np(phi, 0, azimuthal_s, gamma_o, gamma_t);

            sin_theta_op = sin_theta_o * cos_2k_alpha.x + cos_theta_o * sin_2k_alpha.x;
            cos_theta_op = cos_theta_o * cos_2k_alpha.x - sin_theta_o * sin_2k_alpha.x;
            cos_theta_op = abs(cos_theta_op);

            float Mp1 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.y);
            float Np1 = Np(phi, 1, azimuthal_s, gamma_o, gamma_t);

            sin_theta_op = sin_theta_o * cos_2k_alpha.z + cos_theta_o * sin_2k_alpha.z;
            cos_theta_op = cos_theta_o * cos_2k_alpha.z - sin_theta_o * sin_2k_alpha.z;
            cos_theta_op = abs(cos_theta_op);

            float Mp2 = Mp(cos_theta_i, cos_theta_op, sin_theta_i, sin_theta_op, longitudinal_v.z);
            float Np2 = Np(phi, 2, azimuthal_s, gamma_o, gamma_t);

            float MpMax = Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, longitudinal_v.z);
            float NpMax = 1.f / (2.f * M_PIf);

            float4 apPdf = computeApPdf(sigma_a, sin_theta_o, cos_theta_o, cos_gamma_o, eta, h);
            record.pdf = Mp0 * apPdf.x * Np0 + Mp1 * apPdf.y * Np1 + Mp2 * apPdf.z * Np2 + MpMax * apPdf.w * NpMax;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;

            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1 - Sqr(sin_theta_o));
            float phiO = atan2(wo.z, wo.y);

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float4 apPdf = computeApPdf(sigma_a, sin_theta_o, cos_theta_o, cos_gamma_o, eta, h);

            float ap_pdf_array[] = { apPdf.x, apPdf.y, apPdf.z, apPdf.w };

            int p = 0;
            float eps1 = record.sampler->Next();
            for (p = 0; p < 3; ++p) {
                if (eps1 < ap_pdf_array[p])
                    break;
                eps1 -= ap_pdf_array[p];
            }

            float sin_theta_op = 0.f, cos_theta_op = 0.f;
            if (p == 0) {
                sin_theta_op = sin_theta_o * cos_2k_alpha.y - cos_theta_o * sin_2k_alpha.y;
                cos_theta_op = cos_theta_o * cos_2k_alpha.y + sin_theta_o * sin_2k_alpha.y;
            } else if (p == 1) {
                sin_theta_op = sin_theta_o * cos_2k_alpha.x + cos_theta_o * sin_2k_alpha.x;
                cos_theta_op = cos_theta_o * cos_2k_alpha.x - sin_theta_o * sin_2k_alpha.x;
            } else if (p == 2) {
                sin_theta_op = sin_theta_o * cos_2k_alpha.z + cos_theta_o * sin_2k_alpha.z;
                cos_theta_op = cos_theta_o * cos_2k_alpha.z - sin_theta_o * sin_2k_alpha.z;
            } else {
                sin_theta_op = sin_theta_o;
                cos_theta_op = cos_theta_o;
            }

            float eps2 = max(record.sampler->Next(), 1e-5f);

            float vp[] = { longitudinal_v.x, longitudinal_v.y, longitudinal_v.z };

            float cosTheta =
                1.f + vp[p] * log(eps2 + (1.f - eps2) * exp(-2.f / vp[p]));
            float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
            float cosPhi = cos(2 * M_PIf * record.sampler->Next());
            float sin_theta_i = -cosTheta * sin_theta_op + sinTheta * cosPhi * cos_theta_op;
            float cos_theta_i = SafeSqrt(1 - Sqr(sin_theta_i));

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float gamma_t = SafeASin(sin_gamma_t);
            float dphi;
            float sample = record.sampler->Next();
            if (p < 3)
                dphi = Phi(p, gamma_o, gamma_t) + SampleTrimmedLogistic(sample, azimuthal_s, -M_PIf, M_PIf);
            else
                dphi = 2 * M_PIf * sample;

            float phiI = phiO + dphi;
            record.wi = normalize(float3(sin_theta_i, cos_theta_i * cos(phiI), cos_theta_i * sin(phiI)));
            GetPdf(record);
            GetBsdf(record);

            record.sampled_type = EBsdfLobeType::GlossyReflection;
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;

        local_bsdf.h = sampled_tex.x;

        local_bsdf.sigma_a = sigma_a.Sample(sampled_tex);

        local_bsdf.cos_2k_alpha = cos_2k_alpha;
        local_bsdf.sin_2k_alpha = sin_2k_alpha;
        local_bsdf.longitudinal_v = longitudinal_v;
        local_bsdf.azimuthal_s = azimuthal_s;

        local_bsdf.eta = 1.55;

        return local_bsdf;
    }
};

}// namespace Pupil::optix::material