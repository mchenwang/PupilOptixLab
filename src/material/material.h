#pragma once

#include "common/texture.h"
#include "predefine.h"

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

struct Material {
    EMatType type = EMatType::_unknown;
    bool twosided = false;

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