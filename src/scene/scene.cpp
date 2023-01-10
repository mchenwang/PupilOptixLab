#include "scene.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"

#include "common/util.h"

#include <iostream>

namespace {
void LoadIntegrator(const scene::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr) return;
    scene::Integrator *integrator = static_cast<scene::Integrator *>(dst);
    std::string value = obj->GetProperty("max_depth");
    if (!value.empty()) {
        integrator->max_depth = std::stoi(value);
    }
}

void LoadTransform(const scene::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr) return;
    scene::Transform *transform = static_cast<scene::Transform *>(dst);
    if (obj->var_name.compare("to_world") == 0) {
        std::string value = obj->GetProperty("matrix");
        if (!value.empty()) {
            auto matrix_elems = util::Split(value, " ");
            if (matrix_elems.size() == 16) {
                for (int i = 0; auto &&e : matrix_elems) {
                    transform->matrix[i++] = std::stof(e);
                }
            } else if (matrix_elems.size() == 9) {
                for (int i = 0, j = 0; auto &&e : matrix_elems) {
                    transform->matrix[i] = std::stof(e);
                    ++i, ++j;
                    if (j % 3 == 0) ++i;
                }
            } else {
                std::cerr << "warring: transform matrix size is " << matrix_elems.size() << "(must be 9 or 16).\n";
                for (size_t i = 0; i < matrix_elems.size() && i < 16; i++) {
                    transform->matrix[i] = std::stof(matrix_elems[i]);
                }
            }
        } else {
            value = obj->GetProperty("scale");
            if (!value.empty()) {
                auto scale_value = util::Split(value, ",");
                if (scale_value.size() == 3) {
                    transform->Scale(std::stof(scale_value[0]),
                                     std::stof(scale_value[1]),
                                     std::stof(scale_value[2]));
                } else if (scale_value.size() == 1) {
                    float t = std::stof(scale_value[0]);
                    transform->Scale(t, t, t);
                } else {
                    std::cerr << "warring: transform scale value size is " << scale_value.size() << "(must be 1 or 3).\n";
                }
            }

            // value = obj->GetProperty("rotate");
            auto scale_obj = obj->GetUniqueSubObject("scale");
            if (scale_obj) {
                auto axis = scale_obj->GetProperty("axis");
                if (!axis.empty()) {
                    auto axis_value = util::Split(axis, ",");
                    if (axis_value.size() == 3) {
                        auto angle = scale_obj->GetProperty("angle");
                        if (!angle.empty()) {
                            transform->Rotate(
                                std::stof(axis_value[0]),
                                std::stof(axis_value[1]),
                                std::stof(axis_value[2]),
                                std::stof(angle));
                        }
                    } else {
                        std::cerr << "warring: transform rotation axis is " << axis << "(must be a 3d vector).\n";
                    }
                }
            }

            value = obj->GetProperty("translate");
            if (!value.empty()) {
                auto translate_value = util::Split(value, ",");
                if (translate_value.size() == 3) {
                    transform->Translate(std::stof(translate_value[0]),
                                         std::stof(translate_value[1]),
                                         std::stof(translate_value[2]));
                } else {
                    std::cerr << "warring: transform translate value size is " << translate_value.size() << "(must be 3).\n";
                }
            }
        }
    } else if (obj->var_name.compare("to_uv") == 0) {
        std::string value = obj->GetProperty("scale");
        if (!value.empty()) {
            auto scale_value = util::Split(value, ",");
            if (scale_value.size() == 3) {
                transform->Scale(std::stof(scale_value[0]),
                                 std::stof(scale_value[1]),
                                 std::stof(scale_value[2]));
            } else if (scale_value.size() == 1) {
                float t = std::stof(scale_value[0]);
                transform->Scale(t, t, t);
            } else {
                std::cerr << "warring: transform scale value size is " << scale_value.size() << "(must be 1 or 3).\n";
            }
        }
    } else {
        std::cerr << "warring: transform " << obj->var_name << " UNKNOWN.\n";
    }
}

void LoadFilm(const scene::xml::Object *obj, void *dst) noexcept {
    if (obj == nullptr) return;
    scene::Film *film = static_cast<scene::Film *>(dst);
    if (obj->type.compare("hdrfilm")) {
        std::cerr << "warring: film only support hdrfilm.\n";
        return;
    }

    std::string value = obj->GetProperty("width");
    if (!value.empty()) film->w = std::stoi(value);
    value = obj->GetProperty("height");
    if (!value.empty()) film->h = std::stoi(value);
}
}// namespace

namespace scene {
Scene::Scene() noexcept {
    SetXmlObjLoadCallBack(xml::ETag::_integrator, LoadIntegrator);

    SetXmlObjLoadCallBack(xml::ETag::_transform, LoadTransform);

    SetXmlObjLoadCallBack(xml::ETag::_film, LoadFilm);

    SetXmlObjLoadCallBack(
        xml::ETag::_sensor,
        [this](const scene::xml::Object *obj, void *dst) noexcept {
            if (obj == nullptr) return;
            scene::Sensor *sensor = static_cast<scene::Sensor *>(dst);
            if (obj->type.compare("perspective")) {
                std::cerr << "warring: sensor only support perspective.\n";
                return;
            }

            std::string value = obj->GetProperty("fov");
            if (!value.empty()) sensor->fov = std::stof(value);

            auto film_obj = obj->GetUniqueSubObject("film");
            InvokeXmlObjLoadCallBack(film_obj, &sensor->film);

            auto transform_obj = obj->GetUniqueSubObject("transform");
            InvokeXmlObjLoadCallBack(transform_obj, &sensor->transform);
        });
}

void Scene::LoadFromXML(std::string_view path) noexcept {
    xml::Parser parser;
    auto scene_xml_root_obj = parser.LoadFromFile(path);
    for (auto &xml_obj : scene_xml_root_obj->sub_object) {
        switch (xml_obj->tag) {
            case xml::ETag::_integrator:
                InvokeXmlObjLoadCallBack(xml_obj, &integrator);
                break;
            case xml::ETag::_sensor:
                InvokeXmlObjLoadCallBack(xml_obj, &sensor);
                break;
            case xml::ETag::_shape:
                // TODO
                break;
        }
    }
}
}// namespace scene
