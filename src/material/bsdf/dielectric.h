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

    CUDA_HOSTDEVICE float3 GetBsdf(float2 tex) const noexcept {
        return make_float3(0.f);
    }

    CUDA_HOSTDEVICE float GetPdf(float3 wi, float3 wo) const noexcept {
        return 0.f;
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 sampled_tex) const noexcept {
        BsdfSampleRecord ret;

        float eta = wo.z > 0.f ? int_ior / ext_ior : ext_ior / int_ior;
        float3 local_albedo = specular_reflectance.Sample(sampled_tex);
        float3 local_transmittance = specular_transmittance.Sample(sampled_tex);
        float cos_theta_t;
        float fresnel = fresnel::DielectricReflectance(eta, abs(wo.z), cos_theta_t);
        if (xi.x < fresnel) {
            ret.wi = optix_util::Reflect(wo);
            ret.pdf = fresnel;
            ret.f = local_albedo * local_transmittance * fresnel / abs(ret.wi.z);
        } else {
            ret.wi = make_float3(-eta * wo.x, -eta * wo.y, -copysignf(cos_theta_t, wo.z));
            ret.pdf = 1.f - fresnel;
            ret.f = local_albedo * local_transmittance * (1.f - fresnel) / abs(ret.wi.z);
        }

        return ret;
    }
};

}// namespace optix_util::material