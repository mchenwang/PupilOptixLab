#pragma once

#include "common/texture.h"
#include "common/enum.h"

#include <array>

namespace scene {
class Scene;

namespace xml {
struct Object;
}
}// namespace scene

namespace material {
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

// struct PrincipledBSDF {
// };

/// Add a new material
/// 1. add name to PUPIL_RENDER_MATERIAL
/// 2. define the material struct
/// 3. add the new material to Material::union struct
/// 4. add the corresponding optix_material
/// 5. implement material loader and [] method for optix
#define PUPIL_RENDER_MATERIAL \
    diffuse, dielectric, conductor, roughconductor

PUPIL_ENUM_DEFINE(EMatType, PUPIL_RENDER_MATERIAL)
PUPIL_ENUM_STRING_ARRAY(S_MAT_TYPE_NAME, PUPIL_RENDER_MATERIAL)

struct Material {
    EMatType type = EMatType::_unknown;

    union {
        Diffuse diffuse;
        Dielectric dielectric;
        Conductor conductor;
        RoughConductor rough_conductor;
        // PrincipledBSDF principled;
    };

    Material() noexcept {}
};

Material LoadMaterialFromXml(const scene::xml::Object *, scene::Scene *) noexcept;
}// namespace material
