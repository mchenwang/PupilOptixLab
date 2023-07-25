#pragma once

#include "cuda/texture.h"
#include "resource/material.h"
#include "bsdf/bsdf.h"

namespace Pupil::optix::material {
using Pupil::EMatType;

struct Material {
    EMatType type;
    bool twosided;
    union {
        Diffuse diffuse;
        Dielectric dielectric;
        RoughDielectric rough_dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
        Plastic plastic;
        RoughPlastic rough_plastic;
    };

    struct LocalBsdf {
        EMatType type;
        union {
            Diffuse::Local diffuse;
            Dielectric::Local dielectric;
            RoughDielectric::Local rough_dielectric;
            Conductor::Local conductor;
            RoughConductor::Local rough_conductor;
            Plastic::Local plastic;
            RoughPlastic::Local rough_plastic;
        };

#ifdef PUPIL_OPTIX
        CUDA_DEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            optixDirectCall<void, BsdfSamplingRecord &, const Material::LocalBsdf &>(
                ((unsigned int)type - 1) * 2, record, *this);
        }

        CUDA_DEVICE void Eval(BsdfSamplingRecord &record) const noexcept {
            optixDirectCall<void, BsdfSamplingRecord &, const Material::LocalBsdf &>(
                ((unsigned int)type - 1) * 2 + 1, record, *this);
        }
#else
        CUDA_DEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            switch (type) {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(enum_type, attr) \
    case EMatType::##enum_type:                          \
        attr.Sample(record);                             \
        break;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
            }
        }

        CUDA_DEVICE void Eval(BsdfSamplingRecord &record) const noexcept {
            switch (type) {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(enum_type, attr) \
    case EMatType::##enum_type:                          \
        attr.GetBsdf(record);                            \
        attr.GetPdf(record);                             \
        break;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
            }
        }
#endif
        CUDA_DEVICE float3 GetAlbedo() const noexcept {
            switch (type) {
                case EMatType::Diffuse:
                    return diffuse.reflectance;
                case EMatType::Dielectric:
                    return dielectric.specular_reflectance;
                case EMatType::RoughDielectric:
                    return rough_dielectric.specular_reflectance;
                case EMatType::Conductor:
                    return conductor.specular_reflectance;
                case EMatType::RoughConductor:
                    return rough_conductor.specular_reflectance;
                case EMatType::Plastic:
                    return plastic.diffuse_reflectance;
                case EMatType::RoughPlastic:
                    return rough_plastic.diffuse_reflectance;
            }
            return make_float3(0.f);
        }
    };

    CUDA_HOSTDEVICE Material() noexcept {}

#ifndef PUPIL_OPTIX
    void LoadMaterial(Pupil::resource::Material mat) noexcept;
#else
    CUDA_DEVICE LocalBsdf GetLocalBsdf(float2 sampled_tex) const noexcept {
        LocalBsdf local_bsdf;
        local_bsdf.type = type;
        switch (type) {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(enum_type, mat_attr)  \
    case EMatType::##enum_type:                               \
        local_bsdf.mat_attr = mat_attr.GetLocal(sampled_tex); \
        break;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
        }
        return local_bsdf;
    }

    CUDA_DEVICE float3 GetColor(float2 sampled_tex) const noexcept {
        switch (type) {
            case EMatType::Diffuse:
                return diffuse.reflectance.Sample(sampled_tex);
            case EMatType::Dielectric:
                return dielectric.specular_reflectance.Sample(sampled_tex);
            case EMatType::RoughDielectric:
                return rough_dielectric.specular_reflectance.Sample(sampled_tex);
            case EMatType::Conductor:
                return conductor.specular_reflectance.Sample(sampled_tex);
            case EMatType::RoughConductor:
                return rough_conductor.specular_reflectance.Sample(sampled_tex);
            case EMatType::Plastic:
                return plastic.diffuse_reflectance.Sample(sampled_tex);
            case EMatType::RoughPlastic:
                return rough_plastic.diffuse_reflectance.Sample(sampled_tex);
        }
        return make_float3(0.f);
    }
#endif
};
}// namespace Pupil::optix::material