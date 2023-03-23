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
            case EMatType::_plastic:
                ret = plastic.Sample(xi, wo, tex);
                break;
            case EMatType::_roughplastic:
                ret = rough_plastic.Sample(xi, wo, tex);
                break;
        }
        return ret;
    }

    CUDA_HOSTDEVICE BsdfEvalRecord Eval(float3 wi, float3 wo, float2 tex) const noexcept {
        BsdfEvalRecord ret;
        switch (type) {
            case EMatType::_diffuse:
                ret.f = diffuse.GetBsdf(tex, wi, wo);
                ret.pdf = diffuse.GetPdf(wi, wo);
                break;
            case EMatType::_dielectric:
                ret.f = dielectric.GetBsdf(tex, wi, wo);
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
            case EMatType::_plastic:
                ret.f = plastic.GetBsdf(tex, wi, wo);
                ret.pdf = plastic.GetPdf(wi, wo);
                break;
            case EMatType::_roughplastic:
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