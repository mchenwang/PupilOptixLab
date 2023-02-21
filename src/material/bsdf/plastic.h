#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "optix_util/util.h"

namespace optix_util::material {

struct Plastic {
    float int_ior;
    float ext_ior;
    bool nonlinear;
    cuda::Texture diffuse_reflectance;
    cuda::Texture specular_reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
        if (wi.z < 0.f || wo.z < 0.f) return make_float3(0.f);
        return diffuse_reflectance.Sample(tex) * M_1_PIf;
    }

    // CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo, float2 sampled_tex) const noexcept {
    //     if (wi.z < 0.f || wo.z < 0.f) return 0.f;
    //     float3 local_diffuse = diffuse_reflectance.Sample(sampled_tex);
    //     float3 local_specular = specular_reflectance.Sample(sampled_tex);
    //     return optix_util::CosineSampleHemispherePdf(wi);
    // }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z < 0.f || wo.z < 0.f) return 0.f;
        return optix_util::CosineSampleHemispherePdf(wi);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = optix_util::CosineSampleHemisphere(xi.x, xi.y);
        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);

        return ret;
    }
};

}// namespace optix_util::material