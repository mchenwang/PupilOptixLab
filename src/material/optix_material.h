#pragma once

#include "cuda_util/texture.h"
#include "material.h"
#include "bsdf/bsdf.h"

namespace optix_util {
namespace material {
using ::material::EMatType;

struct Material {
    EMatType type;
    bool twosided;
    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
    };

    CUDA_HOSTDEVICE Material() noexcept {}

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
    void LoadMaterial(::material::Material mat) noexcept;
#else
    CUDA_HOSTDEVICE float3 GetColor(float2 tex) const noexcept {
        float3 color;
        switch (type) {
            case EMatType::_diffuse:
                color = diffuse.reflectance.Sample(tex);
                break;
            case EMatType::_dielectric:
                color = dielectric.specular_reflectance.Sample(tex);
                break;
            case EMatType::_conductor:
                color = conductor.specular_reflectance.Sample(tex);
                break;
            case EMatType::_roughconductor:
                color = rough_conductor.specular_reflectance.Sample(tex);
                break;
        }
        return color;
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 tex) const noexcept {
        BsdfSampleRecord ret;
        switch (type) {
            case EMatType::_diffuse:
                ret = diffuse.Sample(xi, wo, tex);
                break;
            case EMatType::_dielectric:
                ret = dielectric.Sample(xi, wo, tex);
                break;
            case EMatType::_conductor:
                ret = conductor.Sample(xi, wo, tex);
                break;
            case EMatType::_roughconductor:
                ret = rough_conductor.Sample(xi, wo, tex);
                break;
        }
        return ret;
    }

    CUDA_HOSTDEVICE BsdfEvalRecord Eval(float3 wi, float3 wo, float2 tex) const noexcept {
        BsdfEvalRecord ret;
        // if (twosided) {
        //     wi.z = abs(wi.z);
        //     wo.z = abs(wo.z);
        // }
        switch (type) {
            case EMatType::_diffuse:
                ret.f = diffuse.GetBsdf(tex, wi, wo);
                ret.pdf = diffuse.GetPdf(wi, wo);
                break;
            case EMatType::_dielectric:
                ret.f = dielectric.GetBsdf(tex);
                ret.pdf = dielectric.GetPdf(wi, wo);
                break;
            case EMatType::_conductor:
                ret.f = conductor.GetBsdf();
                ret.pdf = conductor.GetPdf();
                break;
            case EMatType::_roughconductor:
                ret.f = rough_conductor.GetBsdf(tex, wi, wo);
                ret.pdf = rough_conductor.GetPdf(wi, wo);
                break;
        }
        return ret;
    }
#endif
};
}
}// namespace optix_util::material