#pragma once

#include "gas_manager.h"
#include "util/transform.h"
#include "util/aabb.h"
#include "resource/shape.h"
#include "render/geometry.h"
#include "render/material/optix_material.h"

namespace Pupil::world {
struct RenderObject {
public:
    std::string name;

    GAS *gas = nullptr;
    unsigned int visibility_mask = 1;
    util::Transform transform{};
    util::AABB aabb{};

    optix::Geometry geo{};
    optix::material::Material mat{};

    bool is_emitter = false;
    unsigned int sub_emitters_num = 0;

    RenderObject(const resource::ShapeInstance &, unsigned int v_mask = 1) noexcept;

    ~RenderObject() noexcept;

    void UpdateTransform(const util::Transform &new_transform) noexcept;
    void ApplyTransform(const util::Transform &new_transform) noexcept;

private:
    RenderObject(const RenderObject &) = delete;
    RenderObject &operator=(const RenderObject &) = delete;
};

}// namespace Pupil::world