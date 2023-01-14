#pragma once

namespace util {
struct float3 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
    };

    float3(float xyz = 0.f) noexcept : x(xyz), y(xyz), z(xyz) {}
    float3(float x_, float y_, float z_) noexcept : x(x_), y(y_), z(z_) {}
};
}// namespace util