#pragma once

#include "gas_manager.h"
#include "util/transform.h"
#include "resource/shape.h"

namespace Pupil::world {
struct RenderObject {
    std::string id;

    GAS *gas = nullptr;
    unsigned int visibility_mask = 1;
    util::Transform transform{};

    RenderObject(const resource::Shape *shape, const util::Transform &, std::string_view id = "", unsigned int v_mask = 1) noexcept;
    RenderObject(std::string_view shape_id, const util::Transform &, std::string_view id = "", unsigned int v_mask = 1) noexcept;

    ~RenderObject() noexcept = default;

    RenderObject(const RenderObject &) = delete;
    RenderObject &operator=(const RenderObject &) = delete;

    void UpdateTransform(const util::Transform &new_transform) noexcept;
    void ApplyTransform(const util::Transform &new_transform) noexcept;
};

}// namespace Pupil::world