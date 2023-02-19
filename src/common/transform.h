#pragma once

#include "type.h"

namespace util {
struct Transform {
    float matrix[16]{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    void Translate(float x, float y, float z) noexcept;
    // rotation axis is [ux, uy, uz]
    void Rotate(float ux, float uy, float uz, float angle) noexcept;
    void Scale(float x, float y, float z) noexcept;

    void LookAt(const Float3 &origin, const Float3 &target, const Float3 &up) noexcept;

    static Float3 TransformPoint(const Float3 point, const float *transform_matrix) noexcept;
    static Float3 TransformVector(const Float3 vector, const float *transform_matrix) noexcept;
    static Float3 TransformNormal(const Float3 normal, const float *transform_matrix_inv_t) noexcept;
};
}// namespace util