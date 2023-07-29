#pragma once

#include "shape.h"
#include "emitter.h"
#include "xml/object.h"
#include "static.h"

#include <vector>
#include <string>
#include <functional>
#include <array>
#include <filesystem>

namespace Pupil::resource {

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
    float near_clip = 0.01f;
    float far_clip = 10000.f;
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
    std::vector<ShapeInstance> shape_instances;
    std::vector<Emitter> emitters;

    Scene() noexcept = default;
    ~Scene() noexcept = default;

    void LoadXmlObj(const xml::Object *, void *) noexcept;

    void Reset() noexcept;
    bool LoadFromXML(std::filesystem::path) noexcept;
    bool LoadFromXML(std::string_view, std::string_view root = DATA_DIR) noexcept;
};
}// namespace Pupil::resource