#pragma once

#include "shape.h"
#include "xml/object.h"

#include <vector>
#include <string>
#include <functional>
#include <array>
#include <filesystem>

namespace scene {

/// @param max_depth: maximum depth of ray tracing, default = 1
struct Integrator {
    int max_depth = 1;
};

/// @brief rgb hdr film, only have width and height
struct Film {
    int w = 768;
    int h = 576;
};

/// @brief perspective and right-handed camera
struct Sensor {
    float fov = 90.f;
    util::Transform transform{};
    Film film{};
};

/// integrator: only support path tracing
/// sensor: only support perspective and right-handed camera
/// film: rgb and hdr
class Scene {
public:
    // scene resource file root path
    std::filesystem::path scene_root_path;

    Integrator integrator;
    Sensor sensor;
    std::vector<Shape> shapes;

    using XmlObjectLoadCallBack = std::function<void(const xml::Object *, void *)>;
    std::array<XmlObjectLoadCallBack, (size_t)xml::ETag::_count> xml_obj_load_cbs{};

    Scene() noexcept;

    void LoadFromXML(std::string_view) noexcept;

    template<typename Func>
        requires std::invocable<Func, xml::Object *, void *>
    void SetXmlObjLoadCallBack(xml::ETag tag, Func &&func) noexcept {
        xml_obj_load_cbs[static_cast<unsigned int>(tag)] = std::forward<Func>(func);
    }

    void InvokeXmlObjLoadCallBack(const xml::Object *obj, void *dst) noexcept {
        if (obj == nullptr) return;
        xml_obj_load_cbs[static_cast<unsigned int>(obj->tag)](obj, dst);
    }
};
}// namespace scene