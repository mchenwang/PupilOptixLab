#pragma once

#include "util/texture.h"
#include "predefine.h"

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
    EMatType type = EMatType::_unknown;
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
}// namespace Pupil::material