#include "scene.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"

#include "common/util.h"

#include <iostream>

// using namespace scene;

namespace scene {
Scene::Scene() noexcept {
    SetXmlObjLoadCallBack(
        xml::ETag::_integrator,
        [](const xml::Object *obj, void *dst) {
            Integrator *integrator = static_cast<Integrator *>(dst);
            std::string value = obj->GetProperty("max_depth");
            if (!value.empty()) {
                integrator->max_depth = std::stoi(value);
            }
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_transform,
        [](const xml::Object *obj, void *dst) {
            Transform *transform = static_cast<Transform *>(dst);
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
                        std::cerr << "warring: transform matrix size is " << matrix_elems.size() << ".\n";
                        for (size_t i = 0; i < matrix_elems.size() && i < 16; i++) {
                            transform->matrix[i] = std::stof(matrix_elems[i]);
                        }
                    }
                } else {
                    value = obj->GetProperty("translate");

                    value = obj->GetProperty("rotate");

                    value = obj->GetProperty("scale")
                }
            } else {
                std::cerr << "warring: transform " << obj->var_name << " UNKNOWN.\n";
            }
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_film,
        [](const xml::Object *obj, void *dst) {

        });

    SetXmlObjLoadCallBack(
        xml::ETag::_sensor,
        [](const xml::Object *obj, void *dst) {
        });
}

void Scene::LoadFromXML(std::string_view path) noexcept {
    xml::Parser parser;
    auto scene_xml_root_obj = parser.LoadFromFile(path);
    for (auto &xml_obj : scene_xml_root_obj->sub_object) {
        if (xml_obj->tag == xml::ETag::_integrator) {
        }
        // if (xml_obj->obj_name.compare(TagToString(xml::ETag::_integrator)) == 0) {
        //     // LoadXmlObj(xml_obj, integrator);
        //     integrator_cb(xml_obj);
        // } else if (xml_obj->obj_name.compare(TagToString(xml::ETag::_sensor)) == 0) {
        //     LoadXmlObj(xml_obj, sensor);
        // } else if (xml_obj->obj_name.compare(TagToString(xml::ETag::_shape)) == 0) {
        // }
    }
}
}// namespace scene
