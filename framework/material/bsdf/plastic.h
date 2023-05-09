#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include "fresnel.h"

namespace Pupil::optix::material {

struct Plastic {
    float int_ior;
    float ext_ior;
    bool nonlinear;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    // pretreatment var
    float m_specular_sampling_weight;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
        if (wi.z <= 0.f || wo.z <= 0.f) return make_float3(0.f);

        float3 specular_contribution = make_float3(0.f);
        if (Pupil::optix::IsZero(abs(dot(Pupil::optix::Reflect(wi), wo) - 1.f))) {
            specular_contribution = specular_reflectance.Sample(tex) / wi.z;
        }

        float eta = int_ior / ext_ior;
        float fresnel_i = fresnel::DielectricReflectance(eta, wi.z);
        float fresnel_o = fresnel::DielectricReflectance(eta, wo.z);

        float3 local_albedo = diffuse_reflectance.Sample(tex);
        float3 diffuse_contribution = local_albedo * (1.f - fresnel_o) * (1.f - fresnel_i) *
                                      (1.f / (eta * eta)) * M_1_PIf;
        // float fdr_int = fresnel::DiffuseReflectance(1.f / eta);
        // if (nonlinear) {
        //     diffuse_contribution *= 1.f / (1.f - local_albedo * fdr_int);
        // } else {
        //     diffuse_contribution *= 1.f / (1.f - fdr_int);
        // }

        return diffuse_contribution + specular_contribution;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;
        if (m_specular_sampling_weight < 0.f) return 0.f;
        float specular_prob = m_specular_sampling_weight;
        float diffuse_prob = 1.f - specular_prob;

        if (!Pupil::optix::IsZero(abs(dot(Pupil::optix::Reflect(wi), wo) - 1.f))) {
            specular_prob = 0.f;
        }
        return specular_prob + diffuse_prob * Pupil::optix::CosineSampleHemispherePdf(wi);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;

        if (m_specular_sampling_weight <= 0.f || wo.z <= 0.f) {
            ret.pdf = 0.f;
            return ret;
        }

        if (xi.x < m_specular_sampling_weight) {
            ret.wi = Pupil::optix::Reflect(wo);
            ret.lobe_type = EBsdfLobeType::DeltaReflection;
        } else {
            xi.x = (xi.x - m_specular_sampling_weight) / (1.f - m_specular_sampling_weight);
            ret.wi = Pupil::optix::CosineSampleHemisphere(xi.x, xi.y);
            ret.lobe_type = EBsdfLobeType::DiffuseReflection;
        }

        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);
        return ret;
    }
};

}// namespace Pupil::optix::material