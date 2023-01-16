#include "material.h"
#include "ior.h"
#include "scene/scene.h"
#include "scene/texture.h"
#include "scene/xml/object.h"

#include "common/util.h"

#include <iostream>
#include <array>
#include <functional>

namespace {
using material::EMatType;
using material::Material;

template<EMatType Tag>
struct MaterialLoader {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        std::cout << "warring: unknown bsdf [" << obj->type << "].\n";
        return {};
    }
};

inline void LoadTextureOrRGB(const scene::xml::Object *obj, scene::Scene *scene, std::string_view param_name,
                             util::Texture &param, util::float3 default_value = { 0.f, 0.f, 0.f }) {
    auto [texture, rgb] = obj->GetParameter(param_name);

    if (texture == nullptr && rgb.empty()) {
        param = util::Singleton<scene::TextureManager>::instance()->GetColorTexture(default_value.r, default_value.g, default_value.b);
        return;
    }

    if (texture == nullptr) {
        auto value = util::Split(rgb, ",");
        float r, g, b;
        if (value.size() == 3) {
            r = std::stof(value[0]);
            g = std::stof(value[1]);
            b = std::stof(value[2]);
        } else if (value.size() == 1) {
            r = g = b = std::stof(value[0]);
        } else {
            std::cerr << "warring: rgb size is " << value.size() << "(must be 3 or 1)\n";
        }

        param = util::Singleton<scene::TextureManager>::instance()->GetColorTexture(r, g, b);
    } else {
        scene->InvokeXmlObjLoadCallBack(texture, &param);
    }
}

inline void LoadFloat(const scene::xml::Object *obj, scene::Scene *scene, std::string_view param_name,
                      float &param, float default_value = 0.f) {
    auto value = obj->GetProperty(param_name);
    if (value.empty()) {
        param = default_value;
        return;
    }

    param = std::stof(value);
}

template<>
struct MaterialLoader<EMatType::_diffuse> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_diffuse;
        LoadTextureOrRGB(obj, scene, "reflectance", mat.diffuse.reflectance, { 0.5f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_dielectric> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_dielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.dielectric.int_ior = material::LoadIor(value);
        value = obj->GetProperty("ext_ior");
        mat.dielectric.ext_ior = material::LoadIor(value);
        LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.dielectric.specular_reflectance, { 1.f });
        LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_conductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_conductor;
        LoadTextureOrRGB(obj, scene, "eta", mat.conductor.eta, { 1.f });
        LoadTextureOrRGB(obj, scene, "k", mat.conductor.k, { 1.f });
        LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_roughconductor> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_roughconductor;
        LoadFloat(obj, scene, "alpha", mat.rough_conductor.alpha, 0.1f);
        LoadTextureOrRGB(obj, scene, "eta", mat.rough_conductor.eta, { 1.f });
        LoadTextureOrRGB(obj, scene, "k", mat.rough_conductor.k, { 1.f });
        LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_conductor.specular_reflectance, { 1.f });
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

namespace material {
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
}// namespace material