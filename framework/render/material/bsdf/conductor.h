#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "../fresnel.h"

namespace Pupil::optix::material {

struct Conductor {
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    struct Local {
        float3 eta;
        float3 k;
        float3 specular_reflectance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            record.f = make_float3(0.f);
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            record.pdf = 0.f;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            record.wi = Pupil::optix::Reflect(record.wo);
            record.pdf = 1.f;

            float3 fresnel = fresnel::ConductorReflectance(eta, k, record.wo.z);
            record.f = specular_reflectance * fresnel / abs(record.wi.z);
            record.sampled_type = EBsdfLobeType::DeltaReflection;
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.eta = eta.Sample(sampled_tex);
        local_bsdf.k = k.Sample(sampled_tex);
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material