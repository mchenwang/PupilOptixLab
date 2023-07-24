#pragma once

#include "tag.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace pugi {
class xml_node;
}

namespace Pupil::resource {
namespace xml {
/// @brief xml object's property
/// support integer, string, float, boolean, rgb, (for transform matrix, scale, translate)
struct Property {
    std::string name;
    std::string value;

    Property(std::string_view name, std::string_view value) noexcept : name(name), value(value) {}
};

// support bsdf, emitter, film, integrator, sensor, shape, texture, transform;
// sub obj for transform: lookat, rotate
// non-unique sub obj: bsdf, shape, texture
struct Object {
    const ETag tag;
    std::string obj_name;
    std::string var_name;
    std::string id;
    std::string type;
    std::vector<Property> properties;
    std::vector<Object *> sub_object;

    Object(std::string_view obj_name, std::string_view type, ETag obj_tag = ETag::_unknown) noexcept
        : obj_name(obj_name), type(type), tag(obj_tag) {}

    std::string GetProperty(std::string_view) const noexcept;
    Object *GetUniqueSubObject(std::string_view) const noexcept;
    std::vector<Object *> GetSubObjects(std::string_view) const noexcept;

    std::pair<Object *, std::string> GetParameter(std::string_view) const noexcept;
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
}// namespace Pupil::resource::xml