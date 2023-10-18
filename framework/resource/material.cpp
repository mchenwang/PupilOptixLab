#include "material.h"
#include "render/material/ior.h"
#include "resource/scene.h"
#include "resource/texture.h"
#include "resource/xml/object.h"
#include "resource/xml/util_loader.h"

#include "util/util.h"
#include "util/log.h"

#include <array>
#include <functional>

namespace {
using namespace Pupil;
using resource::Material;

template<EMatType Tag>
struct MaterialLoader {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Pupil::Log::Warn("unknown bsdf [{}].", obj->type);
        return {};
    }
};

template<>
struct MaterialLoader<EMatType::Diffuse> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Diffuse;
        resource::xml::LoadTextureOrRGB(obj, scene, "reflectance", mat.diffuse.reflectance, { 0.5f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Dielectric> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Dielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.dielectric.int_ior = material::LoadDielectricIor(value, 1.5046f);
        value = obj->GetProperty("ext_ior");
        mat.dielectric.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.dielectric.specular_reflectance, { 1.f });
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughDielectric> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::RoughDielectric;
        std::string value = obj->GetProperty("int_ior");
        mat.rough_dielectric.int_ior = material::LoadDielectricIor(value, 1.5046f);
        value = obj->GetProperty("ext_ior");
        mat.rough_dielectric.ext_ior = material::LoadDielectricIor(value, 1.000277f);
        resource::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_dielectric.alpha, { 0.1f });
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_dielectric.specular_reflectance, { 1.f });
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_transmittance", mat.rough_dielectric.specular_transmittance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Conductor> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Conductor;
        auto conductor_mat_name = obj->GetProperty("material");

        util::Float3 eta, k;
        if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
            eta = { 0.f };
            k = { 1.f };
        }
        resource::xml::LoadTextureOrRGB(obj, scene, "eta", mat.conductor.eta, eta);
        resource::xml::LoadTextureOrRGB(obj, scene, "k", mat.conductor.k, k);
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughConductor> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::RoughConductor;
        //resource::xml::LoadFloat(obj, "alpha", mat.rough_conductor.alpha, 0.1f);

        auto conductor_mat_name = obj->GetProperty("material");
        util::Float3 eta, k;
        if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
            eta = { 0.f };
            k = { 1.f };
        }
        resource::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_conductor.alpha, { 0.1f });
        resource::xml::LoadTextureOrRGB(obj, scene, "eta", mat.rough_conductor.eta, eta);
        resource::xml::LoadTextureOrRGB(obj, scene, "k", mat.rough_conductor.k, k);
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_conductor.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Plastic> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
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
        resource::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.plastic.diffuse_reflectance, { 0.5f });
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::RoughPlastic> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
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
        //resource::xml::LoadFloat(obj, "alpha", mat.rough_plastic.alpha, 0.1f);
        resource::xml::LoadTextureOrRGB(obj, scene, "alpha", mat.rough_plastic.alpha, { 0.1f });
        resource::xml::LoadTextureOrRGB(obj, scene, "diffuse_reflectance", mat.rough_plastic.diffuse_reflectance, { 0.5f });
        resource::xml::LoadTextureOrRGB(obj, scene, "specular_reflectance", mat.rough_plastic.specular_reflectance, { 1.f });
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Twosided> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat = LoadMaterialFromXml(obj->GetUniqueSubObject("bsdf"), scene);
        mat.twosided = true;
        return mat;
    }
};

template<>
struct MaterialLoader<EMatType::Hair> {
    Material operator()(const resource::xml::Object *obj, resource::Scene *scene) {
        Material mat{};
        mat.type = EMatType::Hair;
        float beta_m, beta_n, alpha;
        resource::xml::LoadFloat(obj, std::string_view("beta_m"), beta_m, 0.3f);
        resource::xml::LoadFloat(obj, "beta_n", beta_n, 0.3f);
        resource::xml::LoadFloat(obj, "alpha", alpha, 0.f);

        util::Float3 v, sin_2k_alpha, cos_2k_alpha;

        v.x = pow(0.726f * beta_m + 0.812f * pow(beta_m, 2) + 3.7f * pow(beta_m, 20.f), 2);
        v.y = 0.25f * v.x;
        v.z = 4 * v.x;

        float s;
        s = 0.626657069f *
            (0.265f * beta_n + 1.194f * pow(beta_n, 2) + 5.372f * pow(beta_n, 22.f));

        sin_2k_alpha.x = sin(alpha);
        cos_2k_alpha.x = sqrt(1 - pow(sin_2k_alpha.x, 2));
        sin_2k_alpha.y = 2 * cos_2k_alpha.x * sin_2k_alpha.x;
        cos_2k_alpha.y = pow(cos_2k_alpha.x, 2) - pow(sin_2k_alpha.x, 2);
        sin_2k_alpha.z = 2 * cos_2k_alpha.y * sin_2k_alpha.y;
        cos_2k_alpha.z = pow(cos_2k_alpha.y, 2) - pow(sin_2k_alpha.y, 2);

        mat.hair.longitudinal_v = v;
        mat.hair.azimuthal_s = s;
        mat.hair.sin_2k_alpha = sin_2k_alpha;
        mat.hair.cos_2k_alpha = cos_2k_alpha;
        resource::xml::LoadTextureOrRGB(obj, scene, "sigma_a", mat.hair.sigma_a, { 0.06f,0.1f,0.2f });
        return mat;
    }
};

using LoaderType = std::function<Material(const resource::xml::Object *, resource::Scene *)>;

#define MAT_LOADER(mat) MaterialLoader<EMatType::##_##mat>()
#define MAT_LOADER_DEFINE(...)                            \
    const std::array<LoaderType, (size_t)EMatType::Count> \
        S_MAT_LOADER = { MAP_LIST(MAT_LOADER, __VA_ARGS__) };

//MAT_LOADER_DEFINE(PUPIL_RENDER_MATERIAL);
const std::array<LoaderType, (size_t)EMatType::Count> S_MAT_LOADER = {
#define PUPIL_MATERIAL_TYPE_DEFINE(type) MaterialLoader<EMatType::##type>(),
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_DEFINE
    MaterialLoader<EMatType::Twosided>()
};
}// namespace

namespace Pupil::resource {
Material LoadMaterialFromXml(const resource::xml::Object *obj, resource::Scene *scene) noexcept {
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
    return {};
}
}// namespace Pupil::resource