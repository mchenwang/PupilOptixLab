#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include "../fresnel.h"

namespace Pupil::optix::material {

struct Plastic {
    // float int_ior;
    // float ext_ior;
    float eta;
    bool nonlinear;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    // pre compute
    float m_int_fdr;
    // float m_ext_fdr;
    float m_specular_sampling_weight;

    struct Local {
        float eta;
        float int_fdr;
        // float ext_fdr;
        float specular_sampling_weight;
        bool nonlinear;
        float3 diffuse_reflectance;
        float3 specular_reflectance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            record.f = make_float3(0.f);
            if (record.wi.z <= 0.f || record.wo.z <= 0.f) return;

            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);
            float fresnel_i = fresnel::DielectricReflectance(eta, record.wi.z);
            float3 diff = diffuse_reflectance / (1.f - (nonlinear ? diffuse_reflectance * int_fdr : make_float3(int_fdr)));
            record.f = diff * (1.f - fresnel_i) * (1.f - fresnel_o) * optix::CosineSampleHemispherePdf(record.wi) / (eta * eta * record.wi.z);
            // record.f = diff * (1.f - fresnel_i) * (1.f - fresnel_o) * M_1_PIf;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            record.pdf = 0.f;
            if (record.wi.z <= 0.f || record.wo.z <= 0.f) return;

            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);
            float specular_prob = (fresnel_o * specular_sampling_weight) /
                                  (fresnel_o * specular_sampling_weight +
                                   (1 - fresnel_o) * (1.f - specular_sampling_weight));

            record.pdf = optix::CosineSampleHemispherePdf(record.wi) * (1.f - specular_prob);
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            if (record.wo.z <= 0.f) return;
            float fresnel_o = fresnel::DielectricReflectance(eta, record.wo.z);

            float2 xi = record.sampler->Next2();
            float specular_prob = (fresnel_o * specular_sampling_weight) /
                                  (fresnel_o * specular_sampling_weight +
                                   (1 - fresnel_o) * (1.f - specular_sampling_weight));

            if (xi.x < specular_prob) {
                record.sampled_type = EBsdfLobeType::DeltaReflection;
                record.wi = optix::Reflect(record.wo);

                record.f = specular_reflectance * fresnel_o / record.wi.z;
                record.pdf = specular_prob;
            } else {
                record.sampled_type = EBsdfLobeType::DiffuseReflection;
                record.wi = optix::CosineSampleHemisphere(
                    (xi.x - specular_prob) / (1.f - specular_prob), xi.y);

                float fresnel_i = fresnel::DielectricReflectance(eta, record.wi.z);
                float3 diff = diffuse_reflectance / (1.f - (nonlinear ? diffuse_reflectance * int_fdr : make_float3(int_fdr)));
                record.f = diff * (1.f - fresnel_i) * (1.f - fresnel_o) * optix::CosineSampleHemispherePdf(record.wi) / (eta * eta * record.wi.z);
                record.pdf = optix::CosineSampleHemispherePdf(record.wi) * (1.f - specular_prob);
            }
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.eta = eta;
        local_bsdf.nonlinear = nonlinear;
        local_bsdf.int_fdr = m_int_fdr;
        // local_bsdf.ext_fdr = m_ext_fdr;
        local_bsdf.diffuse_reflectance = diffuse_reflectance.Sample(sampled_tex);
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        local_bsdf.specular_sampling_weight = m_specular_sampling_weight;
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material