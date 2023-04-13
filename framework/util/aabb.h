#pragma once

#include "type.h"
#include "transform.h"

namespace Pupil::util {
struct AABB {
    Float3 min;
    Float3 max;

    AABB()
    noexcept : min(std::numeric_limits<float>::max()),
               max(std::numeric_limits<float>::lowest()) {}

    AABB(const Float3 &min_, const Float3 &max_)
    noexcept : min(min_), max(max_) {}

    AABB(const AABB &) = default;

    void Merge(const Float3 &point) noexcept {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }

    void Merge(float x, float y, float z) noexcept {
        min.x = std::min(min.x, x);
        min.y = std::min(min.y, y);
        min.z = std::min(min.z, z);
        max.x = std::max(max.x, x);
        max.y = std::max(max.y, y);
        max.z = std::max(max.z, z);
    }

    void Merge(const AABB &other) noexcept {
        Merge(other.min);
        Merge(other.max);
    }

    void Transform(const Transform &trans) noexcept {
        Float3 vertex[8] = {
            { min.x, min.y, min.z },
            { min.x, min.y, max.z },
            { min.x, max.y, min.z },
            { min.x, max.y, max.z },
            { max.x, min.y, min.z },
            { max.x, min.y, max.z },
            { max.x, max.y, min.z },
            { max.x, max.y, max.z }
        };
        min = { std::numeric_limits<float>::max() };
        max = { std::numeric_limits<float>::lowest() };
        for (auto &&v : vertex) Merge(Transform::TransformPoint(v, trans.matrix));
    }
};
}// namespace Pupil::util