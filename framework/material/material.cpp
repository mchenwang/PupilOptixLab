#include "material.h"
#include "ior.h"
#include "scene/scene.h"
#include "scene/texture.h"
#include "scene/xml/object.h"
#include "scene/xml/util_loader.h"

#include "util/util.h"

#include <iostream>
#include <array>
#include <functional>

namespace {
using namespace Pupil;
using material::EMatType;
using material::Material;

template<EMatType Tag>
struct MaterialLoader {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        std::cout << "warring: unknown bsdf [" << obj->type << "].\n";
        return {};
    }
};

template<>
struct MaterialLoader<EMatType::_diffuse> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_diffuse;
        scene::xml::LoadTextureOrRGB(obj, scene, "reflectance", mat.diffuse.reflectance, { 0.5f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_dielectric> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_dielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.dielectric.int_ior = material::LoadIor(value, 1.5046f);
        value = obj->GetProperty("ext_ior");
        mat.dielectric.ext_ior = material::LoadIor(value, 1.000277f);
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.dielectric.specular_reflectance, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_conductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_conductor;
        scene::xml::LoadTextureOrRGB(obj, scene, "eta", mat.conductor.eta, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "k", mat.conductor.k, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_roughconductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_roughconductor;
        scene::xml::LoadFloat(obj, "alpha", mat.rough_conductor.alpha, 0.1f);
        scene::xml::LoadTextureOrRGB(obj, scene, "eta", mat.rough_conductor.eta, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "k", mat.rough_conductor.k, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_plastic> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_plastic;
        std::string value = obj->GetProperty("int_ior");
        mat.plastic.int_ior = material::LoadIor(value, 1.49f);
        value = obj->GetProperty("ext_ior");
        mat.plastic.ext_ior = material::LoadIor(value, 1.000277f);
        value = obj->GetProperty("nonlinear");
        if (value.compare("true"))
            mat.plastic.nonlinear = true;
        else
            mat.plastic.nonlinear = false;
        scene::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.plastic.diffuse_reflectance, { 0.5f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_roughplastic> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_roughplastic;
        std::string value = obj->GetProperty("int_ior");
        mat.rough_plastic.int_ior = material::LoadIor(value, 1.49f);
        value = obj->GetProperty("ext_ior");
        mat.rough_plastic.ext_ior = material::LoadIor(value, 1.000277f);
        value = obj->GetProperty("nonlinear");
        if (value.compare("true"))
            mat.rough_plastic.nonlinear = true;
        else
            mat.rough_plastic.nonlinear = false;
        scene::xml::LoadFloat(obj, "alpha", mat.rough_plastic.alpha, 0.1f);
        scene::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.rough_plastic.diffuse_reflectance, { 0.5f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_twosided> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat = material::LoadMaterialFromXml(obj->GetUniqueSubObject("bsdf"), scene);
        mat.twosided = true;
        return mat;
    }
};

using LoaderType = std::function<material::Material(const scene::xml::Object *, scene::Scene *)>;

#define MAT_LOADER(mat) MaterialLoader<EMatType::##_##mat>()
#define MAT_LOADER_DEFINE(...)                             \
    const std::array<LoaderType, (size_t)EMatType::_count> \
        S_MAT_LOADER = { MAP_LIST(MAT_LOADER, __VA_ARGS__) };

MAT_LOADER_DEFINE(PUPIL_RENDER_MATERIAL);
}// namespace

namespace Pupil::material {
Material LoadMaterialFromXml(const scene::xml::Object *obj, scene::Scene *scene) noexcept {
    if (obj == nullptr || scene == nullptr) {
        std::cerr << "warring: (LoadMaterialFromXml) obj or scene is null.\n";
        return {};
    }

    for (int i = 0; auto &&name : S_MAT_TYPE_NAME) {
        if (obj->type.compare(name) == 0) {
            return S_MAT_LOADER[i](obj, scene);
        }
        ++i;
    }

    std::cout << "warring: unknown bsdf [" << obj->type << "].\n";
    return material::Material{};
}
}// namespace Pupil::material