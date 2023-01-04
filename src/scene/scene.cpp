#include "scene.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"

using namespace scene;

namespace scene {
Scene::Scene() noexcept {
    SetXmlObjLoadCallBack(
        xml::ETag::_integrator,
        [](const xml::Object *obj, void *dst) {
            Integrator *integrator = static_cast<Integrator *>(dst);
            std::string value = obj->GetProperty("max_depth");
            integrator->max_depth = std::stoi(value);
        });

    SetXmlObjLoadCallBack(
        xml::ETag::_transform,
        [](const xml::Object *obj, void *dst) {
            Transform *transform = static_cast<Transform *>(dst);
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
