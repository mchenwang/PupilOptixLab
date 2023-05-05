#pragma once

#include "cuda/texture.h"
#include "material.h"
#include "bsdf/bsdf.h"

namespace Pupil::optix {
namespace material {
using Pupil::material::EMatType;

struct Material {
    EMatType type;
    bool twosided;
    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
        Plastic plastic;
        RoughPlastic rough_plastic;
    };

    CUDA_HOSTDEVICE Material() noexcept {}

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
    void LoadMaterial(Pupil::material::Material mat) noexcept;
#else
    CUDA_HOSTDEVICE float3 GetColor(float2 tex) const noexcept {
        switch (type) {
            case EMatType::Diffuse:
                return diffuse.reflectance.Sample(tex);
            case EMatType::Dielectric:
                return dielectric.specular_reflectance.Sample(tex);
            case EMatType::Conductor:
                return conductor.specular_reflectance.Sample(tex);
            case EMatType::RoughConductor:
                return rough_conductor.specular_reflectance.Sample(tex);
            case EMatType::Plastic:
                return plastic.diffuse_reflectance.Sample(tex);
            case EMatType::RoughPlastic:
                return rough_plastic.diffuse_reflectance.Sample(tex);
        }
        return make_float3(0.f);
    }

    CUDA_HOSTDEVICE BsdfSampleRecord Sample(float2 xi, float3 wo, float2 tex) const noexcept {
       /* return optixDirectCall<BsdfSampleRecord, Material &, float2, float3, float2>(
            (unsigned int)type, *this, xi, wo, tex);*/
         BsdfSampleRecord ret;
         switch (type) {
             case EMatType::Diffuse:
                 ret = diffuse.Sample(xi, wo, tex);
                 break;
             case EMatType::Dielectric:
                 ret = dielectric.Sample(xi, wo, tex);
                 break;
             case EMatType::Conductor:
                 ret = conductor.Sample(xi, wo, tex);
                 break;
             case EMatType::RoughConductor:
                 ret = rough_conductor.Sample(xi, wo, tex);
                 break;
             case EMatType::Plastic:
                 ret = plastic.Sample(xi, wo, tex);
                 break;
             case EMatType::RoughPlastic:
                 ret = rough_plastic.Sample(xi, wo, tex);
                 break;
         }
         return ret;
    }

    CUDA_HOSTDEVICE BsdfEvalRecord Eval(float3 wi, float3 wo, float2 tex) const noexcept {
        BsdfEvalRecord ret;
        switch (type) {
            case EMatType::Diffuse:
                ret.f = diffuse.GetBsdf(tex, wi, wo);
                ret.pdf = diffuse.GetPdf(wi, wo);
                break;
            case EMatType::Dielectric:
                ret.f = dielectric.GetBsdf(tex, wi, wo);
                ret.pdf = dielectric.GetPdf(wi, wo);
                break;
            case EMatType::Conductor:
                ret.f = conductor.GetBsdf();
                ret.pdf = conductor.GetPdf();
                break;
            case EMatType::RoughConductor:
                ret.f = rough_conductor.GetBsdf(tex, wi, wo);
                ret.pdf = rough_conductor.GetPdf(wi, wo);
                break;
            case EMatType::Plastic:
                ret.f = plastic.GetBsdf(tex, wi, wo);
                ret.pdf = plastic.GetPdf(wi, wo);
                break;
            case EMatType::RoughPlastic:
                ret.f = rough_plastic.GetBsdf(tex, wi, wo);
                ret.pdf = rough_plastic.GetPdf(wi, wo);
                break;
        }
        return ret;
    }
#endif
};
}
}// namespace Pupil::optix::material