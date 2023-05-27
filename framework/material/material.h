#pragma once
#ifdef PUPIL_OPTIX_LAUNCHER_SIDE

#include "util/texture.h"
#include "predefine.h"
#include "optix/pipeline.h"
#include "optix/module.h"

namespace Pupil::scene {
class Scene;

namespace xml {
struct Object;
}
}// namespace Pupil::scene

namespace Pupil::material {
struct Diffuse {
    util::Texture reflectance;
};

struct Dielectric {
    float int_ior;
    float ext_ior;
    util::Texture specular_reflectance;
    util::Texture specular_transmittance;
};

struct Conductor {
    util::Texture eta;
    util::Texture k;
    util::Texture specular_reflectance;
};

struct RoughConductor {
    float alpha;
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
    float alpha;
    float int_ior;
    float ext_ior;
    bool nonlinear;
    // bool sample_visible;
    util::Texture diffuse_reflectance;
    util::Texture specular_reflectance;
};

// struct PrincipledBSDF {
// };

struct Material {
    EMatType type = EMatType::Unknown;
    bool twosided = false;

    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
        Plastic plastic;
        RoughPlastic rough_plastic;
        // PrincipledBSDF principled;
    };

    Material() noexcept {}
};

Material LoadMaterialFromXml(const Pupil::scene::xml::Object *, Pupil::scene::Scene *) noexcept;

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
#include "material_decl.inl"
#undef PUPIL_MATERIAL_ATTR_DEFINE
#undef PUPIL_TO_STRING
#undef _PUPIL_TO_STRING
    };
}
}// namespace Pupil::material

#endif// PUPIL_OPTIX_LAUNCHER_SIDE