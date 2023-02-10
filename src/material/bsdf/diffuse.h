#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "optix_util/util.h"

namespace optix_util::material {

struct Diffuse {
    cuda::Texture reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex) const noexcept {
        return reflectance.Sample(tex) / M_PIf;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z * wo.z < 0.f) return 0.f;
        float3 wh = wi + wo;
        if (optix_util::IsZero(wh)) return 0.f;
        wh = normalize(wh);
        return dot(wh, wi) / M_PIf;
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = optix_util::CosineSampleHemisphere(xi.x, xi.y);
        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex);
    }
};

}// namespace optix_util::material