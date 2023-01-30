#pragma once

#include "cuda_util/texture.h"
#include "material.h"

namespace optix_util {
namespace material {
using ::material::EMatType;

#if defined(__CUDACC__) || defined(__CUDABE__)
#define MATERIAL_LOAD_FUNC(type)
#else
#define MATERIAL_LOAD_FUNC(type) void LoadMaterial(const type &mat) noexcept
#endif

struct Diffuse {
    cuda::Texture reflectance;

    MATERIAL_LOAD_FUNC(::material::Diffuse);
};

struct Dielectric {
    float int_ior;
    float ext_ior;
    cuda::Texture specular_reflectance;
    cuda::Texture specular_transmittance;

    MATERIAL_LOAD_FUNC(::material::Dielectric);
};

struct Conductor {
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    MATERIAL_LOAD_FUNC(::material::Conductor);
};

struct RoughConductor {
    float alpha;
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    MATERIAL_LOAD_FUNC(::material::RoughConductor);
};

struct Material {
    EMatType type;
    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
    };

    CUDA_HOSTDEVICE Material() noexcept {}

#if defined(__CUDACC__) || defined(__CUDABE__)
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
#endif

#if !defined(__CUDACC__) && !defined(__CUDABE__)
    void LoadMaterial(::material::Material mat) noexcept {
        type = mat.type;
        switch (type) {
            case EMatType::_diffuse:
                diffuse.LoadMaterial(mat.diffuse);
                break;
            case EMatType::_dielectric:
                dielectric.LoadMaterial(mat.dielectric);
                break;
            case EMatType::_conductor:
                conductor.LoadMaterial(mat.conductor);
                break;
            case EMatType::_roughconductor:
                rough_conductor.LoadMaterial(mat.rough_conductor);
                break;

                // case new material
        }
    }
#endif
};
}
}// namespace optix_util::material