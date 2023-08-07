#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "../fresnel.h"

namespace Pupil::optix::material {

struct Dielectric {
    float int_ior;
    float ext_ior;
    cuda::Texture specular_reflectance;
    cuda::Texture specular_transmittance;

    struct Local {
        float eta;
        float3 specular_reflectance;
        float3 specular_transmittance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            record.f = make_float3(0.f);
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            record.pdf = 0.f;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float cos_theta_t;
            float fresnel = fresnel::DielectricReflectance(eta, record.wo.z, cos_theta_t);
            if (record.sampler->Next() < fresnel) {
                record.wi = Pupil::optix::Reflect(record.wo);
                record.pdf = fresnel;
                record.f = specular_reflectance * fresnel / abs(record.wi.z);
                record.sampled_type = EBsdfLobeType::DeltaReflection;
            } else {
                record.wi = Pupil::optix::Refract(record.wo, cos_theta_t, eta);
                record.pdf = 1.f - fresnel;
                // ret.f = local_transmittance * (1.f - fresnel);
                float factor = cos_theta_t < 0.f ? 1.f / eta : eta;
                record.f = specular_transmittance * (1.f - fresnel) * factor * factor / abs(record.wi.z);
                record.sampled_type = EBsdfLobeType::DeltaTransmission;
            }
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.eta = int_ior / ext_ior;
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        local_bsdf.specular_transmittance = specular_transmittance.Sample(sampled_tex);
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material