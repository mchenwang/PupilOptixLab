#pragma once

namespace util {
struct Transform {
    float matrix[16]{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    void Translate(float x, float y, float z) noexcept;
    // rotation axis is [ux, uy, uz]
    void Rotate(float ux, float uy, float uz, float angle) noexcept;
    void Scale(float x, float y, float z) noexcept;
};
}// namespace util