#include "material.h"
#include "ior.h"
#include "scene/scene.h"
#include "scene/texture.h"
#include "scene/xml/object.h"
#include "scene/xml/util_loader.h"

#include "util/util.h"
#include "util/log.h"

#include <array>
#include <functional>

namespace {
using namespace Pupil;
using material::EMatType;
using material::Material;

template<EMatType Tag>
struct MaterialLoader {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Pupil::Log::Warn("unknown bsdf [{}].", obj->type);
        return {};
    }
};

template<>
struct MaterialLoader<EMatType::Diffuse> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Diffuse;
        scene::xml::LoadTextureOrRGB(obj, scene, "reflectance", mat.diffuse.reflectance, { 0.5f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Dielectric> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Dielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.dielectric.int_ior = material::LoadDielectricIor(value, 1.5046f);
        value = obj->GetProperty("ext_ior");
        mat.dielectric.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.dielectric.specular_reflectance, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughDielectric> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::RoughDielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.rough_dielectric.int_ior = material::LoadDielectricIor(value, 1.5046f);
        value = obj->GetProperty("ext_ior");
        mat.rough_dielectric.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        scene::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_dielectric.alpha, { 0.1f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_dielectric.specular_reflectance, { 1.f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.rough_dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Conductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Conductor;
        auto conductor_mat_name = obj->GetProperty("material");

        util::Float3 eta, k;
        if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
            eta = { 0.f };
            k = { 1.f };
        }
        scene::xml::LoadTextureOrRGB(obj, scene, "eta", mat.conductor.eta, eta);
        scene::xml::LoadTextureOrRGB(obj, scene, "k", mat.conductor.k, k);
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughConductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::RoughConductor;
        //scene::xml::LoadFloat(obj, "alpha", mat.rough_conductor.alpha, 0.1f);

        auto conductor_mat_name = obj->GetProperty("material");
        util::Float3 eta, k;
        if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
            eta = { 0.f };
            k = { 1.f };
        }
        scene::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_conductor.alpha, { 0.1f });
        scene::xml::LoadTextureOrRGB(obj, scene, "eta", mat.rough_conductor.eta, eta);
        scene::xml::LoadTextureOrRGB(obj, scene, "k", mat.rough_conductor.k, k);
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Plastic> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Plastic;
        std::string value = obj->GetProperty("int_ior");
        mat.plastic.int_ior = material::LoadDielectricIor(value, 1.49f);
        value = obj->GetProperty("ext_ior");
        mat.plastic.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        value = obj->GetProperty("nonlinear");
        if (value.compare("true") == 0)
            mat.plastic.nonlinear = true;
        else
            mat.plastic.nonlinear = false;
        scene::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.plastic.diffuse_reflectance, { 0.5f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughPlastic> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::RoughPlastic;
        std::string value = obj->GetProperty("int_ior");
        mat.rough_plastic.int_ior = material::LoadDielectricIor(value, 1.49f);
        value = obj->GetProperty("ext_ior");
        mat.rough_plastic.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        value = obj->GetProperty("nonlinear");
        if (value.compare("true") == 0)
            mat.rough_plastic.nonlinear = true;
        else
            mat.rough_plastic.nonlinear = false;
        //scene::xml::LoadFloat(obj, "alpha", mat.rough_plastic.alpha, 0.1f);
        scene::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_plastic.alpha, { 0.1f });
        scene::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.rough_plastic.diffuse_reflectance, { 0.5f });
        scene::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Twosided> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat = material::LoadMaterialFromXml(obj->GetUniqueSubObject("bsdf"), scene);
        mat.twosided = true;
        return mat;
    }
};

using LoaderType = std::function<material::Material(const scene::xml::Object *, scene::Scene *)>;

#define MAT_LOADER(mat) MaterialLoader<EMatType::##_##mat>()
#define MAT_LOADER_DEFINE(...)                            \
    const std::array<LoaderType, (size_t)EMatType::Count> \
        S_MAT_LOADER = { MAP_LIST(MAT_LOADER, __VA_ARGS__) };

//MAT_LOADER_DEFINE(PUPIL_RENDER_MATERIAL);
const std::array<LoaderType, (size_t)EMatType::Count> S_MAT_LOADER = {
#define PUPIL_MATERIAL_TYPE_DEFINE(type) MaterialLoader<EMatType::##type>(),
#include "material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_DEFINE
    MaterialLoader<EMatType::Twosided>()
};
}// namespace

namespace Pupil::material {
Material LoadMaterialFromXml(const scene::xml::Object *obj, scene::Scene *scene) noexcept {
    if (obj == nullptr || scene == nullptr) {
        Pupil::Log::Warn("obj or scene is null.\n\tlocation: LoadMaterialFromXml().");
        return {};
    }

    for (int i = 0; auto &&name : S_MAT_TYPE_NAME) {
        if (obj->type.compare(name) == 0) {
            return S_MAT_LOADER[i](obj, scene);
        }
        ++i;
    }

    Pupil::Log::Warn("unknown bsdf [{}]", obj->type);
    return material::Material{};
}
}// namespace Pupil::material