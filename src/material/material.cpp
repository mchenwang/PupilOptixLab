#include "material.h"
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

template<>
struct MaterialLoader<EMatType::_diffuse> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_diffuse;
        auto [texture, rgb] = obj->GetParameter("reflectance");
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

            mat.diffuse.reflectance = util::Singleton<scene::TextureManager>::instance()->GetColorTexture(r, g, b);
        } else {
            scene->InvokeXmlObjLoadCallBack(texture, &mat.diffuse.reflectance);
        }
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::_dielectric> {
    Material operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Material mat{};
        mat.type = EMatType::_dielectric;

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