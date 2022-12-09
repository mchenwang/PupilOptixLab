#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace pugi {
class xml_node;
}

namespace scene {
namespace xml {
/// @brief xml object's property
/// support integer, string, float, boolean, rgb, (for transform matrix, scale, rotate, translate)
struct Property {
    std::string name;
    std::string value;

    Property(std::string_view name, std::string_view value) noexcept : name(name), value(value) {}
};

/// @brief support bsdf, emitter, film, integrator, sensor, shape, texture, transform, lookat(sub obj for transform)
struct Object {
    std::string obj_name;
    std::string var_name;
    std::string id;
    std::string type;
    std::vector<Property> properties;
    std::vector<Object *> sub_object;

    Object(std::string_view obj_name, std::string_view type) noexcept : obj_name(obj_name), type(type) {}

    std::string GetProperty(std::string_view) noexcept;
};

struct GlobalManager {
    Object *current_obj = nullptr;
    std::vector<std::unique_ptr<Object>> objects_pool;
    std::unordered_map<std::string, std::string> global_params;
    std::unordered_map<std::string, Object *> ref_objects_map;

    void AddGlobalParam(std::string, std::string) noexcept;

    void ReplaceDefaultValue(pugi::xml_node *) noexcept;
};
}
}// namespace scene::xml