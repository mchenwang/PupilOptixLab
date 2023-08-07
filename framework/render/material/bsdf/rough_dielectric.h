#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "../ggx.h"

namespace Pupil::optix::material {

struct RoughDielectric {
    float eta;
    cuda::Texture alpha;
    cuda::Texture specular_reflectance;
    cuda::Texture specular_transmittance;

    struct Local {
        float alpha;
        float eta;
        float3 specular_reflectance;
        float3 specular_transmittance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            record.f = make_float3(0.f);
            if (optix::IsZero(record.wo.z)) return;

            float3 wh;
            bool sample_reflect = record.wo.z * record.wi.z > 0.f;
            if (sample_reflect) {
                wh = normalize(record.wo + record.wi);
            } else {
                wh = normalize(record.wo +
                               record.wi * (record.wo.z > 0.f ? eta : 1.f / eta));
            }

            wh = wh * (wh.z > 0.f ? 1.f : -1.f);

            float F = fresnel::DielectricReflectance(eta, dot(record.wo, wh));
            float G = ggx::G(record.wi, record.wo, alpha);
            float D = ggx::D(wh, alpha);
            if (sample_reflect) {
                record.f = specular_reflectance *
                           F * G * D / (4.f * abs(record.wi.z) * abs(record.wo.z));
            } else {
                float _eta = record.wo.z > 0.f ? eta : 1.f / eta;
                float sqrt_denom = dot(record.wo, wh) + _eta * dot(record.wi, wh);
                record.f = specular_transmittance *
                           abs((1.f - F) * D * G * dot(record.wi, wh) * dot(record.wo, wh) /
                               (sqrt_denom * sqrt_denom * record.wi.z * record.wo.z));
            }
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            record.pdf = 0.f;
            bool sample_reflect = record.wo.z * record.wi.z > 0.f;
            float3 wh;
            float dwh_dwo;
            if (sample_reflect) {
                wh = normalize(record.wo + record.wi);
                dwh_dwo = 1.f / (4.f * dot(record.wi, wh));
            } else {
                float _eta = record.wo.z > 0.f ? eta : 1.f / eta;
                wh = normalize(record.wo + record.wi * _eta);
                float sqrt_denom = dot(record.wo, wh) + _eta * dot(record.wi, wh);
                dwh_dwo = (_eta * _eta * dot(record.wi, wh)) / (sqrt_denom * sqrt_denom);
            }

            wh = wh * (wh.z > 0.f ? 1.f : -1.f);
            float3 wo = record.wo * (record.wo.z > 0.f ? 1.f : -1.f);

            float F = fresnel::DielectricReflectance(eta, dot(record.wo, wh));
            record.pdf = abs(ggx::Pdf(wo, wh, alpha) * (sample_reflect ? F : 1.f - F) * dwh_dwo);
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float2 xi = record.sampler->Next2();
            float3 wo = record.wo * (record.wo.z > 0.f ? 1.f : -1.f);
            float3 wh = ggx::Sample(wo, alpha, xi);
            float cos_theta_t = 0.f;
            float F = fresnel::DielectricReflectance(eta, dot(record.wo, wh), cos_theta_t);
            if (record.sampler->Next() < F) {
                record.wi = optix::Reflect(record.wo, wh);
                record.sampled_type = EBsdfLobeType::GlossyReflection;

#ifndef GGX_Sample_Visible_Area
                if (record.wi.z * record.wo.z <= 0.f)
                    return;
#endif
            } else {
                if (optix::IsZero(cos_theta_t)) return;
                record.wi = optix::Refract(record.wo, wh, cos_theta_t, eta);
                record.sampled_type = EBsdfLobeType::GlossyTransmission;
                if (record.wi.z * record.wo.z >= 0.f)
                    return;
            }
            GetPdf(record);
            GetBsdf(record);
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.alpha = alpha.Sample(sampled_tex).x;
        local_bsdf.eta = eta;
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        local_bsdf.specular_transmittance = specular_transmittance.Sample(sampled_tex);
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material