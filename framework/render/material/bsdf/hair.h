#pragma once

#include "cuda/texture.h"
#include "hair_util.h"

namespace Pupil::optix::material {
struct Hair {
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

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;
            float3 wi = record.wi;
            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1.f - Sqr(sin_theta_o));
            float phi_o = atan2(wo.z, wo.y);

            float sin_theta_i = wi.x;
            float cos_theta_i = SafeSqrt(1.f - Sqr(sin_theta_i));
            float phi_i = atan2(wi.z, wi.y);

            float sin_theta_t = sin_theta_o / eta;
            float cos_theta_t = SafeSqrt(1.f - Sqr(sin_theta_t));

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float cos_gamma_t = SafeSqrt(1.f - Sqr(sin_gamma_t));
            float gamma_t = SafeASin(sin_gamma_t);

            float fac = (2.f * cos_gamma_t / cos_theta_t);
            float3 T = optix::Exp(-sigma_a * fac);

            float fresnel = fresnel::DielectricReflectance(eta, cos_theta_o * cos_gamma_o);

            float phi = phi_i - phi_o;
            float3 f = make_float3(0.f);

            float3 sin_theta_op;
            float3 cos_theta_op;

            sin_theta_op = sin_theta_o * cos_2k_alpha - cos_theta_o * sin_2k_alpha;
            cos_theta_op = cos_theta_o * cos_2k_alpha + sin_theta_o * sin_2k_alpha;
            cos_theta_op = make_float3(abs(cos_theta_op.x), abs(cos_theta_op.y), abs(cos_theta_op.z));

            float mp0 = Mp(cos_theta_i, cos_theta_op.y, sin_theta_i, sin_theta_op.y, longitudinal_v.x);
            float np0 = Np(phi, 0, azimuthal_s, gamma_o, gamma_t);
            float3 r_result = mp0 * Ap0(fresnel) * np0;
            f += r_result;

            float mp1 = Mp(cos_theta_i, cos_theta_op.x, sin_theta_i, sin_theta_op.x, longitudinal_v.y);
            float np1 = Np(phi, 1, azimuthal_s, gamma_o, gamma_t);
            float3 tt_result = mp1 * Ap1(fresnel, T) * np1;
            f += tt_result;

            float mp2 = Mp(cos_theta_i, cos_theta_op.z, sin_theta_i, sin_theta_op.z, longitudinal_v.z);
            float np2 = Np(phi, 2, azimuthal_s, gamma_o, gamma_t);
            float3 trt_result = mp2 * Ap2(fresnel, T) * np2;
            f += trt_result;

            float mp_max = Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, longitudinal_v.z);
            float np_max = 1.f / (2.f * M_PIf);
            float3 max_result = mp_max * ApMax(fresnel, T) * np_max;
            f += max_result;

            if (!optix::IsZero(wi.z)) f /= abs(wi.z);

            record.f = f;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;
            float3 wi = record.wi;
            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1.f - Sqr(sin_theta_o));
            float phi_o = atan2(wo.z, wo.y);

            float sin_theta_i = wi.x;
            float cos_theta_i = SafeSqrt(1.f - Sqr(sin_theta_i));
            float phi_i = atan2(wi.z, wi.y);

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float gamma_t = SafeASin(sin_gamma_t);

            float phi = phi_i - phi_o;

            float3 sin_theta_op;
            float3 cos_theta_op;

            sin_theta_op = sin_theta_o * cos_2k_alpha - cos_theta_o * sin_2k_alpha;
            cos_theta_op = cos_theta_o * cos_2k_alpha + sin_theta_o * sin_2k_alpha;
            cos_theta_op = make_float3(abs(cos_theta_op.x), abs(cos_theta_op.y), abs(cos_theta_op.z));

            float mp0 = Mp(cos_theta_i, cos_theta_op.y, sin_theta_i, sin_theta_op.y, longitudinal_v.x);
            float np0 = Np(phi, 0, azimuthal_s, gamma_o, gamma_t);

            float mp1 = Mp(cos_theta_i, cos_theta_op.x, sin_theta_i, sin_theta_op.x, longitudinal_v.y);
            float np1 = Np(phi, 1, azimuthal_s, gamma_o, gamma_t);

            float mp2 = Mp(cos_theta_i, cos_theta_op.z, sin_theta_i, sin_theta_op.z, longitudinal_v.z);
            float np2 = Np(phi, 2, azimuthal_s, gamma_o, gamma_t);

            float mp_max = Mp(cos_theta_i, cos_theta_o, sin_theta_i, sin_theta_o, longitudinal_v.z);
            float np_max = 1.f / (2.f * M_PIf);

            float4 ap_pdf = ComputeApPdf(sigma_a, sin_theta_o, cos_theta_o, cos_gamma_o, eta, h);
            record.pdf = mp0 * ap_pdf.x * np0 + mp1 * ap_pdf.y * np1 + mp2 * ap_pdf.z * np2 + mp_max * ap_pdf.w * np_max;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float3 wo = record.wo;

            float sin_theta_o = wo.x;
            float cos_theta_o = SafeSqrt(1.f - Sqr(sin_theta_o));
            float phi_o = atan2(wo.z, wo.y);

            float gamma_o = asin(h);
            float cos_gamma_o = cos(gamma_o);

            float4 ap_pdf = ComputeApPdf(sigma_a, sin_theta_o, cos_theta_o, cos_gamma_o, eta, h);

            float ap_pdf_array[] = { ap_pdf.x, ap_pdf.y, ap_pdf.z, ap_pdf.w };

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

            float cos_theta = 1.f + vp[p] * log(eps2 + (1.f - eps2) * exp(-2.f / vp[p]));
            float sin_theta = SafeSqrt(1.f - Sqr(cos_theta));
            float cos_phi = cos(2.f * M_PIf * record.sampler->Next());
            float sin_theta_i = -cos_theta * sin_theta_op + sin_theta * cos_phi * cos_theta_op;
            float cos_theta_i = SafeSqrt(1.f - Sqr(sin_theta_i));

            float etap = sqrt(eta * eta - Sqr(sin_theta_o)) / cos_theta_o;
            float sin_gamma_t = h / etap;
            float gamma_t = SafeASin(sin_gamma_t);
            float dphi;
            float sample = record.sampler->Next();
            if (p < 3)
                dphi = Phi(p, gamma_o, gamma_t) + SampleTrimmedLogistic(sample, azimuthal_s, -M_PIf, M_PIf);
            else
                dphi = 2.f * M_PIf * sample;

            float phi_i = phi_o + dphi;
            record.wi = normalize(float3(sin_theta_i, cos_theta_i * cos(phi_i), cos_theta_i * sin(phi_i)));
            GetPdf(record);
            GetBsdf(record);

            record.sampled_type = EBsdfLobeType::GlossyReflection;
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;

        local_bsdf.h = sampled_tex.y;

        local_bsdf.sigma_a = sigma_a.Sample(sampled_tex);

        local_bsdf.cos_2k_alpha = cos_2k_alpha;
        local_bsdf.sin_2k_alpha = sin_2k_alpha;
        local_bsdf.longitudinal_v = longitudinal_v;
        local_bsdf.azimuthal_s = azimuthal_s;

        local_bsdf.eta = 1.55f;

        return local_bsdf;
    }
};

}// namespace Pupil::optix::material