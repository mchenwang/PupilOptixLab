#pragma once

namespace util {
struct Float3 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
    };

    Float3(float xyz = 0.f) noexcept : x(xyz), y(xyz), z(xyz) {}
    Float3(float x_, float y_, float z_) noexcept : x(x_), y(y_), z(z_) {}
};

struct Float4 {
    union {
        struct {
            float x, y, z, w;
        };
        struct {
            float r, g, b, a;
        };
    };

    Float4(float xyzw = 0.f) noexcept : x(xyzw), y(xyzw), z(xyzw), w(xyzw) {}
    Float4(float x_, float y_, float z_, float w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
};

struct Mat4 {
};

}// namespace util