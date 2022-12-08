#pragma once

#include "tag.h"
#include "object.h"

#include "pugixml.hpp"

#include <iostream>
#include <array>
#include <functional>

namespace scene::xml {

using VisitorFunc = std::function<bool(GlobalManager *, pugi::xml_node &)>;

namespace {
template<ETag T>
struct Visitor {
    bool operator()(GlobalManager *global_manager, pugi::xml_node &node) {
        std::cout << node.name() << " skip\n";
        return false;
    }
};

#define IMPL_VISITOR(Tag, code)                                                \
    template<>                                                                 \
    struct Visitor<Tag> {                                                      \
        bool operator()(GlobalManager *global_manager, pugi::xml_node &node) { \
            code                                                               \
        }                                                                      \
    };

// clang-format off
IMPL_VISITOR(ETag::_scene,
    std::cout << "read scene\n";
    auto scene_root = std::make_unique<Object>(node.name(), node.attribute("version").value());
    global_manager->current_obj = scene_root.get();
    global_manager->objects_pool.emplace_back(std::move(scene_root));
    return true;
)

IMPL_VISITOR(ETag::_default,
    global_manager->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    global_manager->AddGlobalParam(name, value);
    return true;
)

IMPL_VISITOR(ETag::_ref,
    global_manager->ReplaceDefaultValue(&node);
    auto id_attr = node.attribute("id");
    if (!id_attr.empty()) {
        if (global_manager->ref_objects_map.find(id_attr.value()) != global_manager->ref_objects_map.end())
            global_manager->current_obj->sub_object.emplace_back(global_manager->ref_objects_map[id_attr.value()]);
    }
    return true;
)

/*<look_at origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>*/
IMPL_VISITOR(ETag::_lookat,
    global_manager->ReplaceDefaultValue(&node);
    auto lookat_obj = std::make_unique<Object>(node.name(), "");
    auto origin = node.attribute("origin");
    lookat_obj->properties.emplace_back(origin.name(), origin.value());
    auto target = node.attribute("target");
    lookat_obj->properties.emplace_back(target.name(), target.value());
    auto up = node.attribute("up");
    lookat_obj->properties.emplace_back(up.name(), up.value());

    if (global_manager->current_obj) {
        global_manager->current_obj->sub_object.emplace_back(lookat_obj.get());
    }
    global_manager->objects_pool.emplace_back(std::move(lookat_obj));
    return true;
)

// clang-format on

#define TAG_VISITOR(tag) Visitor<ETag::##_##tag>()
#define TAG_VISITORS_DEFINE(...)                           \
    std::array<VisitorFunc, 1 + TAG_ARGS_NUM(__VA_ARGS__)> \
        S_TAG_VISITORS = { Visitor<ETag::UNKNOWN>(), MAP_LIST(TAG_VISITOR, __VA_ARGS__) };

TAG_VISITORS_DEFINE(PUPIL_XML_TAGS);

bool PropertyVisitor(GlobalManager *global_manager, pugi::xml_node &node) {
    global_manager->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    std::string value = node.attribute("value").value();
    if (global_manager->current_obj) {
        global_manager->current_obj->properties.emplace_back(name, value);
    }
    return true;
}

bool ObjectVisitor(GlobalManager *global_manager, pugi::xml_node &node) {
    global_manager->ReplaceDefaultValue(&node);
    auto obj = std::make_unique<Object>(node.name(), node.attribute("type").value());
    auto id_attr = node.attribute("id");
    if (!id_attr.empty()) {
        global_manager->ref_objects_map[id_attr.value()] = obj.get();
    }
    auto name_attr = node.attribute("name");
    if (!name_attr.empty()) {
        obj->var_name = name_attr.value();
    }
    if (global_manager->current_obj) {
        global_manager->current_obj->sub_object.emplace_back(obj.get());
    }

    global_manager->current_obj = obj.get();
    global_manager->objects_pool.emplace_back(std::move(obj));
    return true;
}

bool TransformVisitor(GlobalManager *global_manager, pugi::xml_node &node) {
    global_manager->ReplaceDefaultValue(&node);
    std::string value = node.attribute("value").value();
    if (global_manager->current_obj) {
        global_manager->current_obj->properties.emplace_back(node.name(), value);
    }
    return true;
}
}// namespace

[[nodiscard]] bool Visit(ETag tag, GlobalManager *global_manager, pugi::xml_node &node) {
    VisitorFunc Func;
    switch (tag) {
        case scene::xml::ETag::UNKNOWN:
        case scene::xml::ETag::COUNT:
            Func = S_TAG_VISITORS[0];
            break;
        case scene::xml::ETag::_integer:
        case scene::xml::ETag::_string:
        case scene::xml::ETag::_float:
        case scene::xml::ETag::_rgb:
        case scene::xml::ETag::_boolean:
            Func = PropertyVisitor;
            break;
        case scene::xml::ETag::_matrix:
        case scene::xml::ETag::_scale:
        case scene::xml::ETag::_rotate:
        case scene::xml::ETag::_translate:
            Func = TransformVisitor;
            break;
        case scene::xml::ETag::_bsdf:
        case scene::xml::ETag::_emitter:
        case scene::xml::ETag::_film:
        case scene::xml::ETag::_integrator:
        case scene::xml::ETag::_sensor:
        case scene::xml::ETag::_shape:
        case scene::xml::ETag::_texture:
        case scene::xml::ETag::_transform:
            Func = ObjectVisitor;
            break;
        default:
            Func = S_TAG_VISITORS[static_cast<unsigned int>(tag)];
            break;
    }
    return Func(global_manager, node);
}
}// namespace scene::xml