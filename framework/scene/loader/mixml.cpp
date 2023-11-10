#include "mixml.h"
#include "util.h"

#include "resource/mi_xml/parser.h"
#include "resource/mi_xml/xml_object.h"
#include "resource/shape.h"

#include "render/material/ior.h"

#include "util/log.h"

namespace Pupil::resource::mixml {
    bool MixmlSceneLoader::Load(std::filesystem::path path, Scene* scene) noexcept {
        m_scene_root_path = path.parent_path().make_preferred();

        resource::mixml::Parser parser;

        auto scene_xml_root_obj = parser.LoadFromFile(path);
        if (scene_xml_root_obj == nullptr) return false;

        for (auto& xml_obj : scene_xml_root_obj->sub_object) {
            if (!Visit(xml_obj, scene)) return false;
        }

        return true;
    }

    bool MixmlSceneLoader::Visit(void* obj, Scene* scene) noexcept {
        if (obj == nullptr || scene == nullptr) return false;

        auto xml_obj = static_cast<mixml::Object*>(obj);
        switch (xml_obj->tag) {
            case mixml::ETag::Integrator: {
                scene->max_depth = static_cast<unsigned int>(LoadInt(xml_obj, "max_depth", 2));
            } break;
            case mixml::ETag::Sensor: {
                if (xml_obj->type != "perspective") {
                    Pupil::Log::Error("sensor only support perspective.");
                    return false;
                }

                auto& camera = scene->GetCamera();

                camera.SetNearClip(LoadFloat(xml_obj, "near_clip", 0.01f));
                camera.SetFarClip(LoadFloat(xml_obj, "far_clip", 100.f));

                scene->film_w = 1024;
                scene->film_h = 1024;
                if (auto film_obj = xml_obj->GetUniqueSubObject("film");
                    film_obj) {
                    scene->film_w = static_cast<unsigned int>(LoadInt(film_obj, "width", 768));
                    scene->film_h = static_cast<unsigned int>(LoadInt(film_obj, "height", 576));
                }

                camera.SetAspectRatio(static_cast<float>(scene->film_w) / scene->film_h);

                char fov_axis = 'x';
                if (auto value = xml_obj->GetProperty("fov_axis"); !value.empty()) {
                    if (value == "x" || value == "X") {
                        fov_axis = 'x';
                    } else if (value == "y" || value == "Y") {
                        fov_axis = 'y';
                    } else {
                        Pupil::Log::Warn("sensor fov_axis must be x or y.");
                    }
                }

                float fov = LoadFloat(xml_obj, "fov", 90.f);
                if (fov_axis == 'x') {
                    float aspect = static_cast<float>(scene->film_h) / scene->film_w;
                    float radian = fov * 3.14159265358979323846f / 180.f * 0.5f;
                    float t      = std::tan(radian) * aspect;
                    fov          = 2.f * std::atan(t) * 180.f / 3.14159265358979323846f;
                }
                camera.SetFov(fov);

                auto to_world = LoadTransform(xml_obj->GetUniqueSubObject("transform"));

                // Mitsuba 3: +X points left, +Y points up, +Z points view
                // LookAt view matrix: +X points right, +Y points up, +Z points -view
                to_world.matrix.re[0][0] *= -1;
                to_world.matrix.re[1][0] *= -1;
                to_world.matrix.re[2][0] *= -1;
                to_world.matrix.re[0][2] *= -1;
                to_world.matrix.re[1][2] *= -1;
                to_world.matrix.re[2][2] *= -1;

                camera.SetWorldTransform(to_world);
            } break;
            case mixml::ETag::Shape: {
                auto shape = LoadShape(xml_obj);
                if (shape.Get() == nullptr) return false;

                auto material  = LoadMaterial(xml_obj->GetUniqueSubObject("bsdf"));
                auto transform = LoadTransform(xml_obj->GetUniqueSubObject("transform"));

                resource::TextureInstance emit_radiance;
                if (auto emit_obj = xml_obj->GetUniqueSubObject("emitter");
                    emit_obj && emit_obj->type == "area") {
                    emit_radiance = LoadTexture(emit_obj, "radiance", true, {1.f});
                }

                scene->AddInstance(shape->GetName(), shape, transform, material, emit_radiance);
            } break;
            case mixml::ETag::Emitter: {
                auto emitter = LoadEmitter(xml_obj);
                scene->AddEmitter(std::move(emitter));
            } break;
        }

        return true;
    }

    bool MixmlSceneLoader::LoadBool(mixml::Object* xml_obj, std::string_view param_name, bool default_value) noexcept {
        return util::LoadBool(xml_obj->GetProperty(param_name), default_value);
    }

    int MixmlSceneLoader::LoadInt(mixml::Object* xml_obj, std::string_view param_name, int default_value) noexcept {
        return util::LoadInt(xml_obj->GetProperty(param_name), default_value);
    }

    float MixmlSceneLoader::LoadFloat(mixml::Object* xml_obj, std::string_view param_name, float default_value) noexcept {
        return util::LoadFloat(xml_obj->GetProperty(param_name), default_value);
    }

    util::Float3 MixmlSceneLoader::LoadFloat3(mixml::Object* xml_obj, std::string_view param_name, util::Float3 default_value) noexcept {
        return util::LoadFloat3(xml_obj->GetProperty(param_name), default_value);
    }

    util::CountableRef<Shape> MixmlSceneLoader::LoadShape(mixml::Object* xml_obj) noexcept {
        util::CountableRef<Shape> shape;
        if (xml_obj == nullptr) [[unlikely]]
            return shape;

        if (xml_obj->type == "sphere") {
            shape = Sphere::Make(xml_obj->id);
            shape.As<Sphere>()->SetFlipNormal(LoadBool(xml_obj, "flip_normals", false));
            shape.As<Sphere>()->SetCenter(LoadFloat3(xml_obj, "center", util::Float3(0.f)));
            shape.As<Sphere>()->SetRadius(LoadFloat(xml_obj, "radius", 1.f));
        } else if (xml_obj->type == "cube") {
            shape = Cube::Make(xml_obj->id);
            shape.As<Cube>()->SetFlipNormal(LoadBool(xml_obj, "flip_normals", false));
        } else if (xml_obj->type == "rectangle") {
            shape = Rectangle::Make(xml_obj->id);
            shape.As<Rectangle>()->SetFlipNormal(LoadBool(xml_obj, "flip_normals", false));
        } else if (xml_obj->type == "hair") {
            auto filename     = xml_obj->GetProperty("filename");
            auto path         = (m_scene_root_path / filename).make_preferred();
            shape             = CurveHair::Make(path.string(), xml_obj->id);
            Curve::EType mode = Curve::EType::Cubic;
            if (auto value = xml_obj->GetProperty("spline_mode"); value.compare("linear") == 0)
                shape.As<CurveHair>()->SetCurveType(Curve::EType::Linear);
            else if (value.compare("quadratic") == 0)
                shape.As<CurveHair>()->SetCurveType(Curve::EType::Quadratic);
            else if (value.compare("catrom") == 0)
                shape.As<CurveHair>()->SetCurveType(Curve::EType::Catrom);
            else
                shape.As<CurveHair>()->SetCurveType(Curve::EType::Cubic);

            shape.As<CurveHair>()->SetWidth(LoadFloat(xml_obj, "radius", 0.1f),
                                            LoadBool(xml_obj, "tapered", false));
        } else if (xml_obj->type == "obj") {
            auto filename = xml_obj->GetProperty("filename");
            auto path     = (m_scene_root_path / filename).make_preferred();

            auto load_flag = resource::EShapeLoadFlag::None;
            if (LoadBool(xml_obj, "face_normals", false)) {
                load_flag = load_flag | resource::EShapeLoadFlag::GenSmoothNormals;
            }

            shape = TriangleMesh::Make(path.string(), load_flag, xml_obj->id);
            shape.As<TriangleMesh>()->SetFlipNormal(LoadBool(xml_obj, "flip_normals", false));
            shape.As<TriangleMesh>()->SetFlipTexcoord(LoadBool(xml_obj, "flip_tex_coords", true));
        } else [[unlikely]] {
            Log::Warn("Unknown shape type [{}].", xml_obj->type);
        }
        return shape;
    }

    util::CountableRef<Material> MixmlSceneLoader::LoadMaterial(mixml::Object* xml_obj) noexcept {
        if (xml_obj == nullptr) [[unlikely]]
            return Diffuse::Make(util::Float3(0.8f), "");

        util::CountableRef<Material> material;
        if (xml_obj->type == "diffuse") {
            material = Diffuse::Make(xml_obj->id);
            material.As<Diffuse>()->SetReflectance(LoadTexture(xml_obj, "reflectance", true, {0.5f}));
        } else if (xml_obj->type == "dielectric") {
            material = Dielectric::Make(xml_obj->id);
            material.As<Dielectric>()->SetIntIOR(material::LoadDielectricIor(xml_obj->GetProperty("int_ior"), 1.5046f));
            material.As<Dielectric>()->SetExtIOR(material::LoadDielectricIor(xml_obj->GetProperty("ext_ior"), 1.000277f));
            material.As<Dielectric>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
            material.As<Dielectric>()->SetSpecularTransmittance(LoadTexture(xml_obj, "specular_transmittance", true, {1.f}));
        } else if (xml_obj->type == "roughdielectric") {
            material = RoughDielectric::Make(xml_obj->id);
            material.As<RoughDielectric>()->SetIntIOR(material::LoadDielectricIor(xml_obj->GetProperty("int_ior"), 1.5046f));
            material.As<RoughDielectric>()->SetExtIOR(material::LoadDielectricIor(xml_obj->GetProperty("ext_ior"), 1.000277f));
            material.As<RoughDielectric>()->SetAlpha(LoadTexture(xml_obj, "alpha", false, {0.1f}));
            material.As<RoughDielectric>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
            material.As<RoughDielectric>()->SetSpecularTransmittance(LoadTexture(xml_obj, "specular_transmittance", true, {1.f}));
        } else if (xml_obj->type == "conductor") {
            material                        = Conductor::Make(xml_obj->id);
            auto         conductor_mat_name = xml_obj->GetProperty("material");
            util::Float3 eta, k;
            if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
                eta = {0.f};
                k   = {1.f};
            }
            material.As<Conductor>()->SetEta(LoadTexture(xml_obj, "eta", false, eta));
            material.As<Conductor>()->SetK(LoadTexture(xml_obj, "k", false, k));
            material.As<Conductor>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
        } else if (xml_obj->type == "roughconductor") {
            material                        = RoughConductor::Make(xml_obj->id);
            auto         conductor_mat_name = xml_obj->GetProperty("material");
            util::Float3 eta, k;
            if (!material::LoadConductorIor(conductor_mat_name, eta, k)) {
                eta = {0.f};
                k   = {1.f};
            }
            material.As<RoughConductor>()->SetEta(LoadTexture(xml_obj, "eta", false, eta));
            material.As<RoughConductor>()->SetK(LoadTexture(xml_obj, "k", false, k));
            material.As<RoughConductor>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
            material.As<RoughConductor>()->SetAlpha(LoadTexture(xml_obj, "alpha", false, {0.1f}));
        } else if (xml_obj->type == "plastic") {
            material = Plastic::Make(xml_obj->id);
            material.As<Plastic>()->SetIntIOR(material::LoadDielectricIor(xml_obj->GetProperty("int_ior"), 1.49f));
            material.As<Plastic>()->SetExtIOR(material::LoadDielectricIor(xml_obj->GetProperty("ext_ior"), 1.000277f));
            material.As<Plastic>()->SetLinear(!LoadBool(xml_obj, "nonlinear", false));
            material.As<Plastic>()->SetDiffuseReflectance(LoadTexture(xml_obj, "diffuse_reflectance", true, {0.5f}));
            material.As<Plastic>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
        } else if (xml_obj->type == "roughplastic") {
            material = RoughPlastic::Make(xml_obj->id);
            material.As<RoughPlastic>()->SetIntIOR(material::LoadDielectricIor(xml_obj->GetProperty("int_ior"), 1.49f));
            material.As<RoughPlastic>()->SetExtIOR(material::LoadDielectricIor(xml_obj->GetProperty("ext_ior"), 1.000277f));
            material.As<RoughPlastic>()->SetLinear(!LoadBool(xml_obj, "nonlinear", false));
            material.As<RoughPlastic>()->SetDiffuseReflectance(LoadTexture(xml_obj, "diffuse_reflectance", true, {0.5f}));
            material.As<RoughPlastic>()->SetSpecularReflectance(LoadTexture(xml_obj, "specular_reflectance", true, {1.f}));
            material.As<RoughPlastic>()->SetAlpha(LoadTexture(xml_obj, "alpha", false, {0.1f}));
        } else if (xml_obj->type == "hair") {
            material = Hair::Make(xml_obj->id);
            material.As<Hair>()->SetAlpha(LoadFloat(xml_obj, "alpha", 0.f));
            material.As<Hair>()->SetBetaM(LoadFloat(xml_obj, "beta_m", 0.3f));
            material.As<Hair>()->SetBetaN(LoadFloat(xml_obj, "beta_n", 0.3f));
            material.As<Hair>()->SetSigmaA(LoadTexture(xml_obj, "sigma_a", false, {0.06f, 0.1f, 0.2f}));
        } else if (xml_obj->type == "twosided") {
            material = Twosided::Make(xml_obj->id);
            material.As<Twosided>()->SetMaterial(LoadMaterial(xml_obj->GetUniqueSubObject("bsdf")));
        } else [[unlikely]] {
            Pupil::Log::Warn("unknown bsdf [{}].", xml_obj->type);
            material = Diffuse::Make(util::Float3(0.8f), xml_obj->id);
        }

        return material;
    }

    TextureInstance MixmlSceneLoader::LoadTexture(mixml::Object* xml_obj, bool sRGB, util::Float3 default_value) noexcept {
        if (xml_obj == nullptr) [[unlikely]]
            return RGBTexture::Make(default_value, "");

        util::CountableRef<Texture> texture;
        if (xml_obj->type.compare("bitmap") == 0) {
            auto filename = xml_obj->GetProperty("filename");
            auto path     = (m_scene_root_path / filename).make_preferred();

            texture = Bitmap::Make(path.string(), sRGB, xml_obj->id);

            if (auto filter_mode = xml_obj->GetProperty("filter_type");
                filter_mode == "bilinear")
                texture.As<Bitmap>()->SetFilterMode(Texture::EFilterMode::Linear);
            else
                texture.As<Bitmap>()->SetFilterMode(Texture::EFilterMode::Point);

            if (auto address_mode = xml_obj->GetProperty("wrap_mode");
                address_mode == "mirror")
                texture.As<Bitmap>()->SetAddressMode(Texture::EAddressMode::Mirror);
            else if (address_mode == "clamp")
                texture.As<Bitmap>()->SetAddressMode(Texture::EAddressMode::Clamp);
            else
                texture.As<Bitmap>()->SetAddressMode(Texture::EAddressMode::Wrap);

        } else if (xml_obj->type.compare("checkerboard") == 0) {
            texture = CheckerboardTexture::Make(
                LoadFloat3(xml_obj, "color0", {0.4f}),
                LoadFloat3(xml_obj, "color1", {0.2f}),
                xml_obj->id);
        } else [[unlikely]] {
            Log::Warn("unknown texture type [{}].", xml_obj->type);
            return RGBTexture::Make(default_value, xml_obj->id);
        }

        TextureInstance instance(texture);
        instance.SetTransform(LoadTransform(xml_obj->GetUniqueSubObject("transform")));

        return instance;
    }

    TextureInstance MixmlSceneLoader::LoadTexture(mixml::Object* xml_obj, std::string_view param_name, bool sRGB, util::Float3 default_value) noexcept {
        if (xml_obj == nullptr)
            return RGBTexture::Make(default_value, param_name);

        auto [tex_obj, rgb] = xml_obj->GetParameter(param_name);

        if (tex_obj == nullptr && rgb.empty()) [[unlikely]]
            return RGBTexture::Make(default_value, param_name);

        if (tex_obj == nullptr)
            return RGBTexture::Make(util::LoadFloat3(rgb, default_value), param_name);

        return LoadTexture(tex_obj, sRGB, default_value);
    }

    util::Transform MixmlSceneLoader::LoadTransform(mixml::Object* xml_obj) noexcept {
        util::Transform transform;
        if (xml_obj == nullptr) return transform;

        if (xml_obj->var_name == "to_world") {
            if (auto matrix_str = xml_obj->GetProperty("matrix"); !matrix_str.empty()) {
                auto value = util::LoadFloatVector(matrix_str);
                if (value.size() == 16) {
                    std::copy_n(value.data(), 16, transform.matrix.e);
                } else if (value.size() == 9) {
                    for (int i = 0, j = 0; auto&& e : value) {
                        transform.matrix.e[i] = e;
                        ++i, ++j;
                        if (j % 3 == 0) ++i;
                    }
                } else [[unlikely]] {
                    Log::Warn("transform matrix size is {}(must be 9 or 16).", value.size());
                }
            } else if (auto look_at = xml_obj->GetUniqueSubObject("lookat"); look_at) {
                util::Float3 origin = LoadFloat3(look_at, "origin", {1.f, 0.f, 0.f});
                util::Float3 target = LoadFloat3(look_at, "target", {0.f, 0.f, 0.f});
                util::Float3 up     = LoadFloat3(look_at, "up", {0.f, 1.f, 0.f});

                transform.LookAt(origin, target, up);

                // Mitsuba 3: +X points left, +Y points up, +Z points view
                // LookAt Transform: +X points right, +Y points up, +Z points -view
                transform.matrix.re[0][0] *= -1;
                transform.matrix.re[1][0] *= -1;
                transform.matrix.re[2][0] *= -1;
                transform.matrix.re[0][2] *= -1;
                transform.matrix.re[1][2] *= -1;
                transform.matrix.re[2][2] *= -1;
            } else {
                auto scale = LoadFloat3(xml_obj, "scale", {1.f, 1.f, 1.f});
                transform.Scale(scale.x, scale.y, scale.z);
                auto rotate_obj = xml_obj->GetUniqueSubObject("rotate");
                if (rotate_obj) {
                    auto axis  = LoadFloat3(rotate_obj, "axis", {0.f, 0.f, 0.f});
                    auto angle = LoadFloat(rotate_obj, "angle", 0.f);
                    if (angle != 0.f && axis != util::Float3{0.f, 0.f, 0.f})
                        transform.Rotate(axis.x, axis.y, axis.z, angle);
                }
                auto translate = LoadFloat3(xml_obj, "translate", {0.f});
                transform.Translate(translate.x, translate.y, translate.z);
            }
        } else if (xml_obj->var_name == "to_uv") {
            auto scale = LoadFloat3(xml_obj, "scale", {1.f, 1.f, 1.f});
            transform.Scale(scale.x, scale.y, scale.z);
        } else [[unlikely]] {
            Pupil::Log::Warn("transform [{}] UNKNOWN.", xml_obj->var_name);
        }

        return transform;
    }

    std::unique_ptr<Emitter> MixmlSceneLoader::LoadEmitter(mixml::Object* xml_obj) noexcept {
        if (xml_obj == nullptr || xml_obj->type == "area") return nullptr;

        if (xml_obj->type == "envmap") {
            auto            filename = xml_obj->GetProperty("filename");
            auto            path     = (m_scene_root_path / filename).make_preferred();
            TextureInstance radiance;
            radiance.SetTexture(Bitmap::Make(path.string(), true, xml_obj->id));
            radiance.GetTexture().As<Bitmap>()->SetAddressMode(Texture::EAddressMode::Wrap);
            radiance.GetTexture().As<Bitmap>()->SetFilterMode(Texture::EFilterMode::Linear);

            auto emitter = std::make_unique<EnvmapEmitter>(radiance);
            emitter->SetTransform(LoadTransform(xml_obj->GetUniqueSubObject("transform")));
            emitter->SetScale(LoadFloat(xml_obj, "scale", 1.f));
            return emitter;
        } else if (xml_obj->type == "constant") {
            auto emitter = std::make_unique<ConstEmitter>(LoadFloat3(xml_obj, "radiance", {1.f}));
            return emitter;
        } else [[unlikely]] {
            Log::Warn("unknown emitter type [{}].", xml_obj->type);
        }

        return nullptr;
    }
}// namespace Pupil::resource::mixml
