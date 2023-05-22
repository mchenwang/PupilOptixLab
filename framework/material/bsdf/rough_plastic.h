#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"
#include "fresnel.h"
#include "../ggx.h"

namespace Pupil::optix::material {

struct RoughPlastic {
    float int_ior;
    float ext_ior;
    bool nonlinear;
    cuda::Texture alpha;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    // pretreatment var
    float m_specular_sampling_weight;

    struct Local {
        float eta;
        bool nonlinear;
        float alpha;
        float3 diffuse_reflectance;
        float3 specular_reflectance;
        float m_specular_sampling_weight;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            float3 f = make_float3(0.f);
            if (record.wi.z > 0.f && record.wo.z > 0.f)
                f = diffuse_reflectance * M_1_PIf;
            record.f = f;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            float pdf = 0.f;
            if (record.wi.z > 0.f && record.wo.z > 0.f)
                pdf = optix::CosineSampleHemispherePdf(record.wi);
            record.pdf = pdf;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float2 xi = record.sampler->Next2();
            record.wi = Pupil::optix::CosineSampleHemisphere(xi.x, xi.y);
            GetPdf(record);
            GetBsdf(record);
            record.sampled_type = EBsdfLobeType::DiffuseReflection;
        }
    };

    CUDA_HOSTDEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.diffuse_reflectance = diffuse_reflectance.Sample(sampled_tex);
        return local_bsdf;
    }
};

// struct RoughPlastic {
//     float alpha;
//     float int_ior;
//     float ext_ior;
//     bool nonlinear;
//     cuda::Texture diffuse_reflectance;
//     cuda::Texture specular_reflectance;

//     // pretreatment var
//     float m_specular_sampling_weight;

//     CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
//         if (wi.z <= 0.f || wo.z <= 0.f) return make_float3(0.f);
//         float3 wh = wi + wo;
//         if (Pupil::optix::IsZero(wh)) return make_float3(0.f);
//         wh = normalize(wh);
//         float eta = int_ior / ext_ior;
//         float fresnel_o = fresnel::DielectricReflectance(eta, dot(wo, wh));
//         float D = ggx::D(wh, alpha);
//         float G = ggx::G(wo, wi, alpha);

//         float3 local_specular = specular_reflectance.Sample(tex);
//         float3 glossy_contribution = local_specular * fresnel_o * D * G / (4.f * wo.z * wi.z);

//         float fresnel_i = fresnel::DielectricReflectance(eta, wi.z);
//         float3 local_diffuse = diffuse_reflectance.Sample(tex);
//         float3 diffuse_contribution = local_diffuse * (1.f - fresnel_o) * (1.f - fresnel_i) * (1.f / (eta * eta)) * M_1_PIf;

//         return glossy_contribution + diffuse_contribution;
//     }

//     CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
//         if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;
//         if (m_specular_sampling_weight <= 0.f) return 0.f;
//         float glossy_prob = m_specular_sampling_weight;
//         float diffuse_prob = 1.f - glossy_prob;

//         float3 wh = wi + wo;
//         if (Pupil::optix::IsZero(wh)) return 0.f;
//         wh = normalize(wh);
//         glossy_prob *= ggx::D(wh, alpha) / (4.f * dot(wh, wo));
//         return glossy_prob + diffuse_prob * Pupil::optix::CosineSampleHemispherePdf(wi);
//     }

//     CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
//         BsdfSampleRecord ret;
//         if (m_specular_sampling_weight <= 0.f || wo.z <= 0.f) {
//             ret.pdf = 0.f;
//             return ret;
//         }

//         if (xi.x < m_specular_sampling_weight) {
//             xi.x = (m_specular_sampling_weight - xi.x) / m_specular_sampling_weight;
//             float3 wh = ggx::Sample(wo, alpha, xi);
//             ret.wi = Pupil::optix::Reflect(wo, wh);
//             ret.lobe_type = EBsdfLobeType::GlossyReflection;
//         } else {
//             xi.x = (xi.x - m_specular_sampling_weight) / (1.f - m_specular_sampling_weight);
//             ret.wi = Pupil::optix::CosineSampleHemisphere(xi.x, xi.y);
//             ret.lobe_type = EBsdfLobeType::DiffuseReflection;
//         }

//         ret.pdf = GetPdf(ret.wi, wo);
//         ret.f = GetBsdf(sampled_tex, ret.wi, wo);
//         return ret;
//     }
// };

}// namespace Pupil::optix::material