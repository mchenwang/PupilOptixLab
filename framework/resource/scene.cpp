#include "scene.h"
#include "material.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"
#include "xml/util_loader.h"

#include "texture.h"

#include "util/util.h"
#include "util/texture.h"
#include "util/log.h"

#include <filesystem>

using namespace Pupil;

namespace Pupil::resource {
void Scene::Reset() noexcept {
    emitters.clear();
    shape_instances.clear();

    integrator = Integrator{};
    sensor = Sensor{};
}

bool Scene::LoadFromXML(std::filesystem::path file) noexcept {
    Reset();
    scene_root_path = file.parent_path().make_preferred();

    xml::Parser parser;
    auto scene_xml_root_obj = parser.LoadFromFile(file);
    for (auto &xml_obj : scene_xml_root_obj->sub_object) {
        switch (xml_obj->tag) {
            case xml::ETag::_integrator:
                LoadXmlObj(xml_obj, &integrator);
                break;
            case xml::ETag::_sensor:
                LoadXmlObj(xml_obj, &sensor);
                break;
            case xml::ETag::_shape: {
                auto shape_ins = resource::LoadShapeInstanceFromXml(xml_obj, this);
                if (shape_ins.shape) {
                    shape_instances.push_back(shape_ins);
                }
            } break;
            case xml::ETag::_emitter: {
                emitters.push_back({});
                LoadXmlObj(xml_obj, &emitters.back());
                if (emitters.back().type == EEmitterType::Area)
                    emitters.pop_back();
            } break;
        }
    }
    return true;
}

bool Scene::LoadFromXML(std::string_view file_name, std::string_view root) noexcept {
    std::filesystem::path src_root(root);
    std::filesystem::path file = src_root / file_name;

    if (!file.has_extension()) {
        Pupil::Log::Error("scene file format do not support.");
        return false;
    }

    if (file.extension().string() == ".xml") {
        return LoadFromXML(file);
    }

    Pupil::Log::Error("scene file format do not support.");
    return false;
}

void Scene::LoadXmlObj(const xml::Object *xml_obj, void *dst) noexcept {
    if (xml_obj == nullptr || dst == nullptr) return;

    switch (xml_obj->tag) {
        case xml::ETag::_integrator: {
            auto integrator = reinterpret_cast<Integrator *>(dst);
            resource::xml::LoadInt(xml_obj, "max_depth", integrator->max_depth, 1);
        } break;
        case xml::ETag::_transform: {
            xml::LoadTransform(xml_obj, dst);
        } break;
        case xml::ETag::_film: {
            resource::Film *film = static_cast<resource::Film *>(dst);
            if (xml_obj->type.compare("hdrfilm")) {
                Pupil::Log::Warn("film only support hdrfilm.");
                return;
            }

            resource::xml::LoadInt(xml_obj, "width", film->w, 768);
            resource::xml::LoadInt(xml_obj, "height", film->h, 576);
        } break;
        case xml::ETag::_sensor: {
            resource::Sensor *sensor = static_cast<resource::Sensor *>(dst);
            if (xml_obj->type.compare("perspective")) {
                Pupil::Log::Warn("sensor only support perspective.");
                return;
            }

            xml::LoadFloat(xml_obj, "fov", sensor->fov, 90.f);
            xml::LoadFloat(xml_obj, "near_clip", sensor->near_clip, 0.01f);
            xml::LoadFloat(xml_obj, "far_clip", sensor->far_clip, 10000.f);

            auto film_obj = xml_obj->GetUniqueSubObject("film");
            LoadXmlObj(film_obj, &sensor->film);

            auto value = xml_obj->GetProperty("fov_axis");
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

            auto transform_obj = xml_obj->GetUniqueSubObject("transform");
            LoadXmlObj(transform_obj, &sensor->transform);

            // Mitsuba 3: +X points left, +Y points up, +Z points view
            // Pupil Transform: +X points right, +Y points up, +Z points -view
            sensor->transform.matrix.re[0][0] *= -1;
            sensor->transform.matrix.re[1][0] *= -1;
            sensor->transform.matrix.re[2][0] *= -1;
            sensor->transform.matrix.re[0][2] *= -1;
            sensor->transform.matrix.re[1][2] *= -1;
            sensor->transform.matrix.re[2][2] *= -1;
        } break;
        case xml::ETag::_texture: {
            util::Texture *texture = static_cast<util::Texture *>(dst);

            if (xml_obj->type.compare("bitmap") == 0) {
                texture->type = util::ETextureType::Bitmap;

                auto value = xml_obj->GetProperty("filename");
                auto path = (scene_root_path / value).make_preferred();
                *texture = util::Singleton<resource::TextureManager>::instance()->GetTexture(path.string());

                value = xml_obj->GetProperty("filter_type");
                if (value.compare("bilinear") == 0)
                    texture->bitmap.filter_mode = util::ETextureFilterMode::Linear;
                else
                    texture->bitmap.filter_mode = util::ETextureFilterMode::Point;

                value = xml_obj->GetProperty("wrap_mode");
                if (value.compare("repeat") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Wrap;
                else if (value.compare("mirror") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Mirror;
                else if (value.compare("clamp") == 0)
                    texture->bitmap.address_mode = util::ETextureAddressMode::Clamp;
                else// default is repeat
                    texture->bitmap.address_mode = util::ETextureAddressMode::Wrap;

            } else if (xml_obj->type.compare("checkerboard") == 0) {
                texture->type = util::ETextureType::Checkerboard;
                util::Float3 p1, p2;
                xml::LoadFloat3(xml_obj, "color0", p1, { 0.4f });
                xml::LoadFloat3(xml_obj, "color1", p2, { 0.2f });
                *texture = util::Singleton<TextureManager>::instance()->GetCheckerboardTexture(p1, p2);
            } else {
                Pupil::Log::Warn("unknown texture type [{}].", xml_obj->type);
                *texture = util::Singleton<TextureManager>::instance()->GetColorTexture(0.f, 0.f, 0.f);
            }

            auto transform_obj = xml_obj->GetUniqueSubObject("transform");
            LoadXmlObj(transform_obj, &texture->transform);
        } break;
        case xml::ETag::_bsdf: {
            resource::Material *m = static_cast<resource::Material *>(dst);
            *m = resource::LoadMaterialFromXml(xml_obj, this);
        } break;
        // case xml::ETag::_shape: {
        //     Shape **shape_p = static_cast<Shape **>(dst);
        //     *shape_p = resource::LoadShapeFromXml(xml_obj, this);
        // } break;
        case xml::ETag::_emitter: {
            Emitter *emitter = static_cast<Emitter *>(dst);
            if (xml_obj->type.compare("area") == 0) {
                emitter->type = EEmitterType::Area;
                xml::LoadTextureOrRGB(xml_obj, this, "radiance", emitter->area.radiance);
            } else if (xml_obj->type.compare("point") == 0) {
                emitter->type = EEmitterType::Point;
                util::Float3 pos;
                xml::Load3Float(xml_obj, "position", pos);
                emitter->point.pos = pos;

                util::Float3 intensity;
                xml::LoadFloat3(xml_obj, "intensity", intensity);
                emitter->point.intensity = intensity;
            } else if (xml_obj->type.compare("constant") == 0) {
                emitter->type = EEmitterType::ConstEnv;
                util::Float3 color;
                xml::LoadFloat3(xml_obj, "radiance", color);
                emitter->const_env.radiance = color;
            } else if (xml_obj->type.compare("envmap") == 0) {
                emitter->type = EEmitterType::EnvMap;
                xml::LoadFloat(xml_obj, "scale", emitter->env_map.scale, 1.f);
                auto value = xml_obj->GetProperty("filename");
                auto path = (scene_root_path / value).make_preferred();
                emitter->env_map.radiance = util::Singleton<resource::TextureManager>::instance()->GetTexture(path.string());
                emitter->env_map.radiance.bitmap.filter_mode = util::ETextureFilterMode::Linear;
                emitter->env_map.radiance.bitmap.address_mode = util::ETextureAddressMode::Wrap;

                auto transform_obj = xml_obj->GetUniqueSubObject("transform");
                emitter->env_map.transform = {};
                LoadXmlObj(transform_obj, &emitter->env_map.transform);

            } else {
                Pupil::Log::Warn("unknown emitter type [{}].", xml_obj->type);
            }
        } break;
    }
}
}// namespace Pupil::resource
