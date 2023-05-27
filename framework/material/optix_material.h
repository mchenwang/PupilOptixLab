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

    CUDA_HOSTDEVICE void Sample(BsdfSampleRecord &ret, float2 xi, float3 wo, float2 tex) const noexcept {
        optixDirectCall<void, BsdfSampleRecord &, const Material &, float2, float3, float2>(
            ((unsigned int)type - 1) * 2, ret, *this, xi, wo, tex);
    }

    CUDA_HOSTDEVICE void Eval(BsdfEvalRecord &ret, float3 wi, float3 wo, float2 tex) const noexcept {
        optixDirectCall<void, BsdfEvalRecord &, const Material &, float3, float3, float2>(
            ((unsigned int)type - 1) * 2 + 1, ret, *this, wi, wo, tex);
    }
#endif
};
}
}// namespace Pupil::optix::material