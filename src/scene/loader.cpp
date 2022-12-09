#include "scene.h"
#include "xml/parser.h"
#include "xml/object.h"
#include "xml/tag.h"

namespace {

}

namespace scene {
void Scene::LoadFromXML(std::string_view path) noexcept {
    xml::Parser parser;
    auto scene_xml_root_obj = parser.LoadFromFile(path);
    for (auto &xml_obj : scene_xml_root_obj->sub_object) {
        if (xml_obj->obj_name.compare(TagToString(xml::ETag::_integrator)) == 0) {
            std::string value = xml_obj->GetProperty("max_depth");
            integrator.max_depth = std::stoi(value);
        } else if (xml_obj->obj_name.compare(TagToString(xml::ETag::_sensor)) == 0) {
        } else if (xml_obj->obj_name.compare(TagToString(xml::ETag::_shape)) == 0) {
        }
    }
}
}// namespace scene
