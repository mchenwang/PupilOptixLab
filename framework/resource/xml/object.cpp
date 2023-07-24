#include "object.h"
#include "pugixml.hpp"

namespace Pupil::resource::xml {
void GlobalManager::AddGlobalParam(std::string name, std::string value) noexcept {
    global_params[name] = value;
}

void GlobalManager::ReplaceDefaultValue(pugi::xml_node *node) noexcept {
    for (auto attr : node->attributes()) {
        std::string a_value = attr.value();
        if (a_value.find('$') == std::string::npos)
            continue;
        for (auto &[p_name, p_value] : global_params) {
            size_t pos = 0;
            std::string temp_name = "$" + p_name;
            while ((pos = a_value.find(temp_name, pos)) != std::string::npos) {
                a_value.replace(pos, temp_name.length(), p_value);
                pos += p_value.length();
            }
        }
        attr.set_value(a_value.c_str());
    }
}

std::string Object::GetProperty(std::string_view property_name) const noexcept {
    for (auto &p : properties) {
        if (p.name.compare(property_name) == 0) {
            return p.value;
        }
    }
    return "";
}

Object *Object::GetUniqueSubObject(std::string_view sub_object_name) const noexcept {
    for (auto so : sub_object) {
        if (so->obj_name.compare(sub_object_name) == 0) {
            return so;
        }
    }
    return nullptr;
}

std::vector<Object *> Object::GetSubObjects(std::string_view sub_object_name) const noexcept {
    std::vector<Object *> ret{};
    for (auto so : sub_object) {
        if (so->obj_name.compare(sub_object_name) == 0) {
            ret.emplace_back(so);
        }
    }
    return ret;
}

std::pair<Object *, std::string> Object::GetParameter(std::string_view target_name) const noexcept {
    for (auto so : sub_object) {
        if (so->var_name.compare(target_name) == 0) {
            return { so, "" };
        }
    }

    return { nullptr, GetProperty(target_name) };
}
}// namespace Pupil::resource::xml
