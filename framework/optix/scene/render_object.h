#pragma once

#include "mesh.h"

namespace Pupil::optix {
struct RenderObject {
    std::string id;

    OptixTraversableHandle gas_handle = 0;
    unsigned int visibility_mask = 1;
    util::Transform transform{};

    RenderObject(EMeshEntityType, void *, util::Transform, std::string_view id = "", unsigned int v_mask = 1) noexcept;
    ~RenderObject() noexcept = default;

    RenderObject(const RenderObject &) = delete;
    RenderObject &operator=(const RenderObject &) = delete;

    void UpdateTransform(const util::Transform &new_transform) noexcept;
    void ApplyTransform(const util::Transform &new_transform) noexcept;
};

}// namespace Pupil::optix