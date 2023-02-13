#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "optix_util/util.h"

namespace optix_util::material {

struct Diffuse {
    cuda::Texture reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex) const noexcept {
        return reflectance.Sample(tex) * M_1_PIf;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        return optix_util::UniformSampleHemispherePdf(wi);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = optix_util::UniformSampleHemisphere(xi.x, xi.y);
        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex);
        return ret;
    }
};

}// namespace optix_util::material