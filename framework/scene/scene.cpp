#include "scene.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"
#include "xml/util_loader.h"

#include "texture.h"

#include "util/util.h"
#include "util/texture.h"
#include "util/log.h"

#include "material/material.h"

#include <filesystem>

using namespace Pupil;
namespace {
void LoadIntegrator(const scene::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr || dst == nullptr) return;
    scene::Integrator *integrator = static_cast<scene::Integrator *>(dst);
    scene::xml::LoadInt(obj, "max_depth", integrator->max_depth, 1);
}

void LoadFilm(const scene::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr || dst == nullptr) return;
    scene::Film *film = static_cast<scene::Film *>(dst);
    if (obj->type.compare("hdrfilm")) {
        Pupil::Log::Warn("film only support hdrfilm.");
        return;
    }

    scene::xml::LoadInt(obj, "width", film->w, 768);
    scene::xml::LoadInt(obj, "height", film->h, 576);
}
}// namespace

namespace Pupil::scene {
Scene::Scene() noexcept {
    SetXmlObjLoadCallBack(xml::ETag::_integrator, LoadIntegrator);

    SetXmlObjLoadCallBack(xml::ETag::_transform, scene::xml::LoadTransform);

    SetXmlObjLoadCallBack(xml::ETag::_film, LoadFilm);

    SetXmlObjLoadCallBack(
        xml::ETag::_sensor,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr || dst == nullptr) return;
            scene::Sensor *sensor = static_cast<scene::Sensor *>(dst);
            if (obj->type.compare("perspective")) {
                Pupil::Log::Warn("sensor only support perspective.");
                return;
            }

            xml::LoadFloat(obj, "fov", sensor->fov, 90.f);
            xml::LoadFloat(obj, "near_clip", sensor->near_clip, 0.01f);
            xml::LoadFloat(obj, "far_clip", sensor->far_clip, 10000.f);

            auto film_obj = obj->GetUniqueSubObject("film");
            InvokeXmlObjLoadCallBack(film_obj, &sensor->film);

            auto value = obj->GetProperty("fov_axis");
            char fov_axis = 'x';
            if (!value.empty()) {
                if (value.compare("x") == 0 || value.compare("X") == 0) {
                    fov_axis = 'x';
                } else if (value.compare("y") == 0 || value.compare("Y") == 0) {
                    fov_axis = 'y';
                } else {
                    Pupil::Log::Warn("sensor fov_axis must be x or y.");
                }
            }

            if (fov_axis == 'x') {
                float aspect = static_cast<float>(sensor->film.h) / static_cast<float>(sensor->film.w);
                float radian = sensor->fov * 3.14159265358979323846f / 180.f * 0.5f;
                float t = std::tan(radian) * aspect;
                sensor->fov = 2.f * std::atan(t) * 180.f / 3.14159265358979323846f;
            }

            auto transform_obj = obj->GetUniqueSubObject("transform");
            InvokeXmlObjLoadCallBack(transform_obj, &sensor->transform);

            // Mitsuba 3: +X points left, +Y points up, +Z points view
            // Pupil Transform: +X points right, +Y points up, +Z points -view
            sensor->transform.matrix.re[0][0] *= -1;
            sensor->transform.matrix.re[1][0] *= -1;
            sensor->transform.matrix.re[2][0] *= -1;
            sensor->transform.matrix.re[0][2] *= -1;
            sensor->transform.matrix.re[1][2] *= -1;
            sensor->transform.matrix.re[2][2] *= -1;
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_texture,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr || dst == nullptr) return;
            util::Texture *texture = static_cast<util::Texture *>(dst);

            if (obj->type.compare("bitmap") == 0) {
                texture->type = util::ETextureType::Bitmap;

                auto value = obj->GetProperty("filename");
                auto path = (scene_root_path / value).make_preferred();
                *texture = util::Singleton<scene::TextureManager>::instance()->GetTexture(path.string());

                value = obj->GetProperty("filter_type");
                if (value.compare("bilinear") == 0)
                    texture->bitmap.filter_mode = util::ETextureFilterMode::Linear;
                else
                    texture->bitmap.filter_mode = util::ETextureFilterMode::Point;

                value = obj->GetProperty("wrap_mode");
                if (value.compare("repeat") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Wrap;
                else if (value.compare("mirror") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Mirror;
                else if (value.compare("clamp") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Clamp;
                else// default is repeat
                    texture->bitmap.address_mode = util::ETextureAddressMode::Wrap;

            } else if (obj->type.compare("checkerboard") == 0) {
                texture->type = util::ETextureType::Checkerboard;
                util::Float3 p1, p2;
                xml::LoadFloat3(obj, "color0", p1, { 0.4f });
                xml::LoadFloat3(obj, "color1", p2, { 0.2f });
                *texture = util::Singleton<TextureManager>::instance()->GetCheckerboardTexture(p1, p2);
            } else {
                Pupil::Log::Warn("unknown texture type [{}].", obj->type);
                *texture = util::Singleton<TextureManager>::instance()->GetColorTexture(0.f, 0.f, 0.f);
            }

            auto transform_obj = obj->GetUniqueSubObject("transform");
            InvokeXmlObjLoadCallBack(transform_obj, &texture->transform);
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_bsdf,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr || dst == nullptr) return;
            material::Material *m = static_cast<material::Material *>(dst);
            *m = material::LoadMaterialFromXml(obj, this);
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_shape,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr || dst == nullptr) return;
            Shape *shape = static_cast<Shape *>(dst);

            *shape = scene::LoadShapeFromXml(obj, this);
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_emitter,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr || dst == nullptr) return;
            Emitter *emitter = static_cast<Emitter *>(dst);

            if (obj->type.compare("area") == 0) {
                emitter->type = EEmitterType::Area;
                xml::LoadTextureOrRGB(obj, this, "radiance", emitter->area.radiance);
            } else if (obj->type.compare("constant") == 0) {
                emitter->type = EEmitterType::ConstEnv;
                util::Float3 color;
                xml::LoadFloat3(obj, "radiance", color);
                emitter->const_env.radiance = color;
            } else if (obj->type.compare("envmap") == 0) {
                // TODO:
            } else {
                Pupil::Log::Warn("unknown emitter type [{}].", obj->type);
            }
        });
}

void Scene::Reset() noexcept {
    emitters.clear();
    shapes.clear();

    integrator = Integrator{};
    sensor = Sensor{};
    aabb = util::AABB{};
}

void Scene::LoadFromXML(std::filesystem::path file) noexcept {
    Reset();
    scene_root_path = file.parent_path().make_preferred();

    xml::Parser parser;
    auto scene_xml_root_obj = parser.LoadFromFile(file);
    for (auto &xml_obj : scene_xml_root_obj->sub_object) {
        switch (xml_obj->tag) {
            case xml::ETag::_integrator:
                InvokeXmlObjLoadCallBack(xml_obj, &integrator);
                break;
            case xml::ETag::_sensor:
                InvokeXmlObjLoadCallBack(xml_obj, &sensor);
                break;
            case xml::ETag::_shape:
                //Shape shape;
                shapes.push_back({});
                InvokeXmlObjLoadCallBack(xml_obj, &shapes.back());
                // Pupil::Log::Info("shape [{}] AABB: min[{:.3f},{:.3f},{:.3f}], max[{:.3f},{:.3f},{:.3f}]",
                //                  xml_obj->id,
                //                  shapes.back().aabb.min.x, shapes.back().aabb.min.y, shapes.back().aabb.min.z,
                //                  shapes.back().aabb.max.x, shapes.back().aabb.max.y, shapes.back().aabb.max.z);
                aabb.Merge(shapes.back().aabb);
                break;
            case xml::ETag::_emitter:
                //Emitter emitter;
                emitters.push_back({});
                InvokeXmlObjLoadCallBack(xml_obj, &emitters.back());
                if (emitters.back().type == EEmitterType::Area)
                    emitters.pop_back();
                break;
        }
    }
    Pupil::Log::Info("scene AABB: min[{:.3f},{:.3f},{:.3f}], max[{:.3f},{:.3f},{:.3f}]",
                     aabb.min.x, aabb.min.y, aabb.min.z,
                     aabb.max.x, aabb.max.y, aabb.max.z);
}

void Scene::LoadFromXML(std::string_view file_name, std::string_view root) noexcept {
    std::filesystem::path src_root(root);
    std::filesystem::path file = src_root / file_name;
    LoadFromXML(file);
}
}// namespace Pupil::scene
