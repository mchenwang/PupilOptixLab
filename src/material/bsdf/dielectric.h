#pragma once

#include "predefine.h"
#include "cuda_util/texture.h"
#include "fresnel.h"

namespace optix_util::material {

struct Dielectric {
    float int_ior;
    float ext_ior;
    cuda::Texture specular_reflectance;
    cuda::Texture specular_transmittance;

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex, float3 wi, float3 wo) const noexcept {
        return make_float3(0.f);
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        return 0.f;
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;

        // float eta = wo.z > 0.f ? int_ior / ext_ior : ext_ior / int_ior;
        float eta = int_ior / ext_ior;
        float cos_theta_t;
        float fresnel = fresnel::DielectricReflectance(eta, wo.z, cos_theta_t);
        if (xi.x < fresnel) {
            ret.wi = optix_util::Reflect(wo);
            ret.pdf = fresnel;
            float3 local_specular = specular_reflectance.Sample(sampled_tex);
            ret.f = local_specular * fresnel / abs(ret.wi.z);
            ret.lobe_type = EBsdfLobeType::DeltaReflection;
        } else {
            // ret.wi = make_float3(-eta * wo.x, -eta * wo.y, -copysignf(cos_theta_t, wo.z));
            ret.wi = optix_util::Refract(wo, cos_theta_t, eta);
            ret.pdf = 1.f - fresnel;
            // ret.f = local_transmittance * (1.f - fresnel);
            float factor = cos_theta_t < 0.f ? 1.f / eta : eta;
            float3 local_transmittance = specular_transmittance.Sample(sampled_tex);
            ret.f = local_transmittance * (1.f - fresnel) * factor * factor / abs(ret.wi.z);
            ret.lobe_type = EBsdfLobeType::DeltaTransmission;
        }

        return ret;
    }
};

}// namespace optix_util::material