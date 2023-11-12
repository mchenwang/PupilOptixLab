#pragma once

#include "material/predefine.h"
#include "material/bsdf/bsdf.h"

namespace Pupil::optix {
    using Pupil::EMatType;

    struct Material {
        EMatType type;
        bool     twosided;
        union {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Type, attr) material::Type attr;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
        };

        CUDA_HOSTDEVICE Material() {}

        struct LocalBsdf {
            EMatType type;
            union {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Type, attr) material::Type## ::Local attr;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
            };

#ifdef PUPIL_OPTIX
            CUDA_DEVICE void Sample(BsdfSamplingRecord& record) const noexcept {
                switch (type) {
#define PUPIL_MATERIAL_TYPE_ATTR_DEFINE(enum_type, attr) \
    case EMatType::##enum_type:                          \
        attr.Sample(record);                             \
        break;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
                }
            }

            CUDA_DEVICE void Eval(BsdfSamplingRecord& record) const noexcept {
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

            CUDA_DEVICE float3 GetAlbedo() const noexcept {
                switch (type) {
#define PUPIL_MATERIAL_ALBEDO_DEFINE(enum_type, attr, albedo) \
    case EMatType::##enum_type:                               \
        return attr.albedo;
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_ALBEDO_DEFINE
                }
                return make_float3(0.f);
            }
#endif
        };

#ifdef PUPIL_OPTIX
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
#endif
    };
}// namespace Pupil::optix