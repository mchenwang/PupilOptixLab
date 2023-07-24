#pragma once

#include "tag.h"
#include "object.h"

#include "pugixml.hpp"

#include "util/log.h"

#include <array>
#include <functional>

namespace Pupil::resource::xml {

using VisitorFunc = std::function<bool(GlobalManager *, pugi::xml_node &)>;

namespace {
inline bool PropertyVisitor(ETag tag, GlobalManager *global_manager, pugi::xml_node &node) {
    global_manager->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    if (name.empty()) name = node.name();
    std::string value = node.attribute("value").value();
    if (global_manager->current_obj) {
        global_manager->current_obj->properties.emplace_back(name, value);
    }
    return true;
}

inline bool ObjectVisitor(ETag tag, GlobalManager *global_manager, pugi::xml_node &node) {
    global_manager->ReplaceDefaultValue(&node);
    auto obj = std::make_unique<Object>(node.name(), node.attribute("type").value(), tag);
    auto id_attr = node.attribute("id");
    if (!id_attr.empty()) {
        obj->id = id_attr.value();
        global_manager->ref_objects_map[obj->id] = obj.get();
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

// inline bool TransformVisitor(ETag tag, GlobalManager *global_manager, pugi::xml_node &node) {
//     global_manager->ReplaceDefaultValue(&node);
//     std::string value = node.attribute("value").value();
//     if (global_manager->current_obj) {
//         global_manager->current_obj->properties.emplace_back(node.name(), value);
//     }
//     return true;
// }

inline bool XYZValuePropertyVisitor(ETag tag, GlobalManager *global_manager, pugi::xml_node &node,
                                    std::string_view default_x, std::string_view default_y, std::string_view default_z) {
    global_manager->ReplaceDefaultValue(&node);
    std::string name = node.attribute("name").value();
    if (name.empty()) name = node.name();
    std::string value = node.attribute("value").value();
    if (value.empty()) {
        std::string x = node.attribute("x").value();
        std::string y = node.attribute("y").value();
        std::string z = node.attribute("z").value();

        if (x.empty()) x = default_x;
        if (y.empty()) y = default_y;
        if (z.empty()) z = default_z;
        value = x + "," + y + "," + z;
    }
    if (global_manager->current_obj) {
        global_manager->current_obj->properties.emplace_back(name, value);
    }
    return true;
}

template<ETag T>
struct Visitor {
    bool operator()(GlobalManager *global_manager, pugi::xml_node &node) {
        Pupil::Log::Warn("XML Node [{}] skip", node.name());
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
    auto scene_root = std::make_unique<Object>(node.name(), node.attribute("version").value(), ETag::_scene);
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

/*<lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>*/
IMPL_VISITOR(ETag::_lookat,
    global_manager->ReplaceDefaultValue(&node);
    auto lookat_obj = std::make_unique<Object>(node.name(), "", ETag::_lookat);
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

// <rotate value="0.701, 0.701, 0.701" angle="180"/>
// <rotate y="1" angle="45"/>
IMPL_VISITOR(ETag::_rotate,
    global_manager->ReplaceDefaultValue(&node);
    auto rotate_obj = std::make_unique<Object>(node.name(), "", ETag::_rotate);
    std::string axis{};
    if (!node.attribute("value").empty()) {
        axis = node.attribute("value").value();
    } else if (!node.attribute("x").empty()) {
        axis = "1, 0, 0";
    } else if (!node.attribute("y").empty()) {
        axis = "0, 1, 0";
    } else if (!node.attribute("z").empty()) {
        axis = "0, 0, 1";
    }
    rotate_obj->properties.emplace_back("axis", axis);

    auto angle = node.attribute("angle");
    rotate_obj->properties.emplace_back(angle.name(), angle.value());

    if (global_manager->current_obj) {
        global_manager->current_obj->sub_object.emplace_back(rotate_obj.get());
    }
    global_manager->objects_pool.emplace_back(std::move(rotate_obj));
    return true;
)

// <scale value="5"/>
// <scale value="2, 1, -1"/>
// <scale x="4" y="2"/>
IMPL_VISITOR(ETag::_scale,      return XYZValuePropertyVisitor(ETag::_scale,     global_manager, node, "1", "1", "1");)

// <point name="center" value="1,1,1"/>
// <point name="center" x="1" y="0" z="0"/>
IMPL_VISITOR(ETag::_point,      return XYZValuePropertyVisitor(ETag::_point,     global_manager, node, "0", "0", "0");)

// <translate x="1" y="0" z="0"/>
IMPL_VISITOR(ETag::_translate,  return XYZValuePropertyVisitor(ETag::_translate, global_manager, node, "0", "0", "0");)

IMPL_VISITOR(ETag::_integer,    return PropertyVisitor(ETag::_integer,  global_manager, node);)
IMPL_VISITOR(ETag::_string,     return PropertyVisitor(ETag::_string,   global_manager, node);)
IMPL_VISITOR(ETag::_float,      return PropertyVisitor(ETag::_float,    global_manager, node);)
IMPL_VISITOR(ETag::_rgb,        return PropertyVisitor(ETag::_rgb,      global_manager, node);)
IMPL_VISITOR(ETag::_boolean,    return PropertyVisitor(ETag::_boolean,  global_manager, node);)
IMPL_VISITOR(ETag::_matrix,     return PropertyVisitor(ETag::_matrix,   global_manager, node);)

IMPL_VISITOR(ETag::_bsdf,       return ObjectVisitor(ETag::_bsdf,       global_manager, node);)
IMPL_VISITOR(ETag::_emitter,    return ObjectVisitor(ETag::_emitter,    global_manager, node);)
IMPL_VISITOR(ETag::_film,       return ObjectVisitor(ETag::_film,       global_manager, node);)
IMPL_VISITOR(ETag::_integrator, return ObjectVisitor(ETag::_integrator, global_manager, node);)
IMPL_VISITOR(ETag::_sensor,     return ObjectVisitor(ETag::_sensor,     global_manager, node);)
IMPL_VISITOR(ETag::_shape,      return ObjectVisitor(ETag::_shape,      global_manager, node);)
IMPL_VISITOR(ETag::_texture,    return ObjectVisitor(ETag::_texture,    global_manager, node);)
IMPL_VISITOR(ETag::_transform,  return ObjectVisitor(ETag::_transform,  global_manager, node);)

// clang-format on

#define TAG_VISITOR(tag) Visitor<ETag::##_##tag>()
#define TAG_VISITORS_DEFINE(...)                                   \
    std::array<VisitorFunc, 1 + PUPIL_MACRO_ARGS_NUM(__VA_ARGS__)> \
        S_TAG_VISITORS = { Visitor<ETag::_unknown>(), MAP_LIST(TAG_VISITOR, __VA_ARGS__) };

TAG_VISITORS_DEFINE(PUPIL_XML_TAGS);

}// namespace

[[nodiscard]] bool Visit(ETag tag, GlobalManager *global_manager, pugi::xml_node &node) {
    return S_TAG_VISITORS[static_cast<unsigned int>(tag)](global_manager, node);
}
}// namespace Pupil::resource::xml