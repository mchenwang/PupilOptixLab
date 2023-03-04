#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "ggx.h"

namespace optix_util::material {

struct RoughConductor {
    float alpha;
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 local_tex, float3 in, float3 out) const noexcept {
        if (optix_util::IsZero(in.z) || optix_util::IsZero(out.z)) return make_float3(0.f);
        float3 wh = in + out;
        if (optix_util::IsZero(wh)) return make_float3(0.f);
        wh = normalize(wh);

        float3 local_eta = eta.Sample(local_tex);
        float3 local_k = k.Sample(local_tex);
        float3 local_albedo = specular_reflectance.Sample(local_tex);

        float IoH = dot(in, wh);
        float3 fresnel = fresnel::ConductorReflectance(local_eta, local_k, IoH);
        float3 f = ggx::D(wh, alpha) * fresnel * ggx::G(in, out, alpha) / (4.f * in.z * out.z);
        return local_albedo * f;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z * wo.z < 0.f) return 0.f;
        float3 wh = wi + wo;
        if (optix_util::IsZero(wh)) return 0.f;
        wh = normalize(wh);
        return ggx::Pdf(wo, wh, alpha) / (4.f * dot(wo, wh));
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = optix_util::Reflect(wo, ggx::Sample(wo, alpha, xi));
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);
        ret.pdf = GetPdf(ret.wi, wo);
        return ret;
    }
};

}// namespace optix_util::material