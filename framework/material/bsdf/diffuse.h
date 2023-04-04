#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"

namespace Pupil::optix::material {

struct Diffuse {
    cuda::Texture reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
        if (wi.z <= 0.f || wo.z <= 0.f) return make_float3(0.f);
        return reflectance.Sample(tex) * M_1_PIf;
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        if (wi.z <= 0.f || wo.z <= 0.f) return 0.f;
        return Pupil::optix::CosineSampleHemispherePdf(wi);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = Pupil::optix::CosineSampleHemisphere(xi.x, xi.y);
        ret.pdf = GetPdf(ret.wi, wo);
        ret.f = GetBsdf(sampled_tex, ret.wi, wo);
        ret.lobe_type = EBsdfLobeType::DiffuseReflection;

        return ret;
    }
};

}// namespace Pupil::optix::material