#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "fresnel.h"

namespace Pupil::optix::material {

struct Conductor {
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    CUDA_HOSTDEVICE float3 GetBsdf() const noexcept {
        return make_float3(0.f);
    }

    CUDA_HOSTDEVICE float GetPdf() const noexcept {
        return 0.f;
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;
        ret.wi = Pupil::optix::Reflect(wo);
        ret.pdf = 1.f;

        float3 local_eta = eta.Sample(sampled_tex);
        float3 local_k = k.Sample(sampled_tex);
        float3 local_albedo = specular_reflectance.Sample(sampled_tex);

        float3 fresnel = fresnel::ConductorReflectance(local_eta, local_k, wo.z);
        ret.f = local_albedo * fresnel / abs(ret.wi.z);
        ret.lobe_type = EBsdfLobeType::DeltaReflection;
        return ret;
    }
};

}// namespace Pupil::optix::material