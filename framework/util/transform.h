#pragma once

#include "type.h"

namespace Pupil::util {
struct Transform {
    Mat4 matrix;

    Transform() noexcept
        : matrix(1.f, 0.f, 0.f, 0.f,
                 0.f, 1.f, 0.f, 0.f,
                 0.f, 0.f, 1.f, 0.f,
                 0.f, 0.f, 0.f, 1.f) {}
    Transform(const Mat4 &m) noexcept : matrix(m) {}

    void Translate(float x, float y, float z) noexcept;
    // rotation axis is [ux, uy, uz]
    void Rotate(float ux, float uy, float uz, float angle) noexcept;
    void Scale(float x, float y, float z) noexcept;

    void LookAt(const Float3 &origin, const Float3 &target, const Float3 &up) noexcept;

    static Float3 TransformPoint(const Float3 point, const Mat4 &transform_matrix) noexcept;
    static Float3 TransformVector(const Float3 vector, const Mat4 &transform_matrix) noexcept;
    static Float3 TransformNormal(const Float3 normal, const Mat4 &transform_matrix_inv_t) noexcept;
};
}// namespace Pupil::util