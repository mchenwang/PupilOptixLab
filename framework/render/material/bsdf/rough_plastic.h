#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include "../fresnel.h"
#include "../ggx.h"

namespace Pupil::optix::material {

struct RoughPlastic {
    float eta;
    bool nonlinear;
    cuda::Texture alpha;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    // pretreatment var
    float m_specular_sampling_weight;
    float m_int_fdr;

    struct Local {
        float eta;
        float int_fdr;
        float specular_sampling_weight;
        float alpha;
        bool nonlinear;
        float3 diffuse_reflectance;
        float3 specular_reflectance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            record.f = make_float3(0.f);
            if (record.wi.z <= 0.f || record.wo.z <= 0.f) return;

            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);

            float3 wh = normalize(record.wi + record.wo);
            record.f = specular_reflectance *
                       fresnel::DielectricReflectance(eta, dot(wh, record.wo)) * ggx::D(wh, alpha) *
                       ggx::G(record.wi, record.wo, alpha) / (4.f * record.wo.z * record.wi.z);

            float fresnel_i = fresnel::DielectricReflectance(eta, record.wi.z);
            float3 diff = diffuse_reflectance / (1.f - (nonlinear ? diffuse_reflectance * int_fdr : make_float3(int_fdr)));
            record.f += diff * (1.f - fresnel_i) * (1.f - fresnel_o) * M_1_PIf / (eta * eta);
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            record.pdf = 0.f;
            if (record.wi.z <= 0.f || record.wo.z <= 0.f) return;

            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);
            float specular_prob = (fresnel_o * specular_sampling_weight) /
                                  (fresnel_o * specular_sampling_weight +
                                   (1 - fresnel_o) * (1.f - specular_sampling_weight));
            float diffuse_prob = 1.f - specular_prob;

            float3 wh = normalize(record.wi + record.wo);
            record.pdf = specular_prob * ggx::Pdf(record.wo, wh, alpha) /
                         (4.f * dot(record.wi, wh));
            record.pdf += diffuse_prob * optix::CosineSampleHemispherePdf(record.wi);
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            record.wi = make_float3(0.f);
            if (record.wo.z <= 0.f) return;

            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);
            float specular_prob = (fresnel_o * specular_sampling_weight) /
                                  (fresnel_o * specular_sampling_weight +
                                   (1 - fresnel_o) * (1.f - specular_sampling_weight));

            float2 xi = record.sampler->Next2();
            if (xi.y < specular_prob) {
                xi.y /= specular_prob;
                float3 wh = ggx::Sample(record.wo, alpha, xi);
                record.wi = optix::Reflect(record.wo, wh);
                record.sampled_type = EBsdfLobeType::GlossyReflection;
            } else {
                xi.y = (xi.y - specular_prob) / (1.f - specular_prob);
                record.wi = optix::CosineSampleHemisphere(xi.x, xi.y);
                record.sampled_type = EBsdfLobeType::DiffuseReflection;
            }
            GetPdf(record);
            GetBsdf(record);
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.eta = eta;
        local_bsdf.nonlinear = nonlinear;
        local_bsdf.int_fdr = m_int_fdr;
        local_bsdf.alpha = alpha.Sample(sampled_tex).x;
        local_bsdf.diffuse_reflectance = diffuse_reflectance.Sample(sampled_tex);
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        local_bsdf.specular_sampling_weight = m_specular_sampling_weight;
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material