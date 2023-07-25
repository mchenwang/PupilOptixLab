#pragma once
#ifndef PUPIL_OPTIX

#include "util/texture.h"
#include "render/material/predefine.h"
#include "optix/pipeline.h"
#include "optix/module.h"

namespace Pupil::resource {
class Scene;

namespace xml {
struct Object;
}

struct Diffuse {
    util::Texture reflectance;
};

struct Dielectric {
    float int_ior;
    float ext_ior;
    util::Texture specular_reflectance;
    util::Texture specular_transmittance;
};

struct RoughDielectric {
    float int_ior;
    float ext_ior;
    util::Texture alpha;
    util::Texture specular_reflectance;
    util::Texture specular_transmittance;
};

struct Conductor {
    util::Texture eta;
    util::Texture k;
    util::Texture specular_reflectance;
};

struct RoughConductor {
    util::Texture alpha;
    util::Texture eta;
    util::Texture k;
    util::Texture specular_reflectance;
};

struct Plastic {
    float int_ior;
    float ext_ior;
    bool nonlinear;
    util::Texture diffuse_reflectance;
    util::Texture specular_reflectance;
};

struct RoughPlastic {
    float int_ior;
    float ext_ior;
    bool nonlinear;
    util::Texture alpha;
    util::Texture diffuse_reflectance;
    util::Texture specular_reflectance;
};

// struct PrincipledBSDF {
// };

struct Material {
    EMatType type = EMatType::Unknown;
    bool twosided = false;

    union {
        Diffuse diffuse{};
        Dielectric dielectric;
        RoughDielectric rough_dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
        Plastic plastic;
        RoughPlastic rough_plastic;
    };

    Material() noexcept {}
};

Material LoadMaterialFromXml(const Pupil::resource::xml::Object *, Pupil::resource::Scene *) noexcept;

inline auto GetMaterialProgramDesc() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto mat_module_ptr = module_mngr->GetModule(optix::EModuleBuiltinType::Material);
    return std::array{
#define _PUPIL_TO_STRING(var) #var
#define PUPIL_TO_STRING(var) _PUPIL_TO_STRING(var)
#define PUPIL_MATERIAL_ATTR_DEFINE(attr)                            \
    optix::CallableProgramDesc{                                     \
        .module_ptr = mat_module_ptr,                               \
        .cc_entry = nullptr,                                        \
        .dc_entry = PUPIL_TO_STRING(PUPIL_MAT_SAMPLE_CALL(attr)),   \
    },                                                              \
        optix::CallableProgramDesc{                                 \
            .module_ptr = mat_module_ptr,                           \
            .cc_entry = nullptr,                                    \
            .dc_entry = PUPIL_TO_STRING(PUPIL_MAT_EVAL_CALL(attr)), \
        },
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_ATTR_DEFINE
#undef PUPIL_TO_STRING
#undef _PUPIL_TO_STRING
    };
}
}// namespace Pupil::resource

#endif