#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "ggx.h"

namespace Pupil::optix::material {

struct RoughConductor {
    float alpha;
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 local_tex, float3 wi, float3 wo) const noexcept {
        if (Pupil::optix::IsZero(wi.z) || Pupil::optix::IsZero(wo.z)) return make_float3(0.f);
        float3 wh = wi + wo;
        if (Pupil::optix::IsZero(wh)) return make_float3(0.f);
        wh = normalize(wh);

        float3 local_eta = eta.Sample(local_tex);
        float3 local_k = k.Sample(local_tex);
        float3 local_albedo = specular_reflectance.Sample(local_tex);

        float IoH = dot(wi, wh);
        float3 fresnel = fresnel::ConductorReflectance(local_eta, local_k, IoH);
        float3 f = ggx::D(wh, alpha) * fresnel * ggx::G(wi, wo, alpha) / (4.f * wi.z * wo.z);
        return local_albedo * f;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z * wo.z <= 0.f) return 0.f;
        float3 wh = wi + wo;
        if (Pupil::optix::IsZero(wh)) return 0.f;
        wh = normalize(wh);
        return ggx::Pdf(wo, wh, alpha) / (4.f * dot(wo, wh));
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = Pupil::optix::Reflect(wo, ggx::Sample(wo, alpha, xi));
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);
        ret.pdf = GetPdf(ret.wi, wo);
        ret.lobe_type = EBsdfLobeType::DiffuseReflection;
        return ret;
    }
};

}// namespace Pupil::optix::material