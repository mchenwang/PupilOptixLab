#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "optix_util/util.h"
#include "fresnel.h"
#include "../ggx.h"

namespace optix_util::material {

struct RoughPlastic {
    float alpha;
    float int_ior;
    float ext_ior;
    bool nonlinear;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    // pretreatment var
    float m_specular_sampling_weight;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
        if (wi.z < 0.f || wo.z < 0.f) return make_float3(0.f);
        float3 wh = wi + wo;
        if (optix_util::IsZero(wh)) return make_float3(0.f);
        wh = normalize(wh);
        float eta = wo.z > 0.f ? int_ior / ext_ior : ext_ior / int_ior;
        float fresnel_o = fresnel::DielectricReflectance(eta, wo.z);
        float D = ggx::D(wh, alpha);
        float G = ggx::G(wo, wi, alpha);

        float3 local_specular = specular_reflectance.Sample(tex);
        float3 glossy_contribution = local_specular * fresnel_o * D * G / (4.f * wo.z * wi.z);

        float fresnel_i = fresnel::DielectricReflectance(eta, wi.z);
        float3 local_diffuse = diffuse_reflectance.Sample(tex);
        float3 diffuse_contribution = local_diffuse * (1.f - fresnel_o) * (1.f - fresnel_i) * (1.f / (eta * eta)) * M_1_PIf;

        return glossy_contribution + diffuse_contribution;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z < 0.f || wo.z < 0.f) return 0.f;
        if (m_specular_sampling_weight < 0.f) return 0.f;
        float glossy_prob = m_specular_sampling_weight;
        float diffuse_prob = 1.f - glossy_prob;

        float3 wh = wi + wo;
        if (optix_util::IsZero(wh)) return 0.f;
        wh = normalize(wh);
        glossy_prob = ggx::D(wh, alpha) / (4.f * dot(wh, wo));
        return glossy_prob + diffuse_prob * optix_util::CosineSampleHemispherePdf(wi);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        if (m_specular_sampling_weight < 0.f || wo.z < 0.f) {
            ret.pdf = 0.f;
            return ret;
        }

        if (xi.x < m_specular_sampling_weight) {
            xi.x = (m_specular_sampling_weight - xi.x) / m_specular_sampling_weight;
            float3 wh = ggx::Sample(wo, alpha, xi);
            ret.wi = optix_util::Reflect(wo, wh);
        } else {
            xi.x = (xi.x - m_specular_sampling_weight) / (1.f - m_specular_sampling_weight);
            ret.wi = optix_util::CosineSampleHemisphere(xi.x, xi.y);
        }

        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);
        return ret;
    }
};

}// namespace optix_util::material