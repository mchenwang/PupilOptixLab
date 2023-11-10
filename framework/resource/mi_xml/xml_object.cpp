#include "xml_object.h"

namespace Pupil::resource::mixml {
    std::string Object::GetProperty(std::string_view property_name) const noexcept {
        for (auto& p : properties) {
            if (p.name.compare(property_name) == 0) {
                return p.value;
            }
        }
        return "";
    }

    Object* Object::GetUniqueSubObject(std::string_view sub_object_name) const noexcept {
        for (auto so : sub_object) {
            if (so->obj_name.compare(sub_object_name) == 0) {
                return so;
            }
        }
        return nullptr;
    }

    std::vector<Object*> Object::GetSubObjects(std::string_view sub_object_name) const noexcept {
        std::vector<Object*> ret{};
        for (auto so : sub_object) {
            if (so->obj_name.compare(sub_object_name) == 0) {
                ret.emplace_back(so);
            }
        }
        return ret;
    }

    std::pair<Object*, std::string> Object::GetParameter(std::string_view target_name) const noexcept {
        for (auto so : sub_object) {
            if (so->var_name.compare(target_name) == 0) {
                return {so, ""};
            }
        }

        return {nullptr, GetProperty(target_name)};
    }
}// namespace Pupil::resource::mixml
