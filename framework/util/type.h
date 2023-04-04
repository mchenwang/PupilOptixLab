#pragma once

#include <string>
#include <DirectXMath.h>

namespace Pupil::util {
struct Float3 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
        float e[3];
    };

    constexpr Float3(float xyz = 0.f) noexcept : x(xyz), y(xyz), z(xyz) {}
    constexpr Float3(float x_, float y_, float z_) noexcept : x(x_), y(y_), z(z_) {}

    inline Float3 operator+(const Float3 &t) const noexcept { return { x + t.x, y + t.y, z + t.z }; }
    inline Float3 operator-(const Float3 &t) const noexcept { return { x - t.x, y - t.y, z - t.z }; }
    inline Float3 operator*(float t) const noexcept { return { x * t, y * t, z * t }; }
    inline Float3 operator/(float t) const noexcept { return *this * (1.f / t); }

    inline Float3 &operator+=(const Float3 &t) noexcept {
        *this = *this + t;
        return *this;
    }
    inline Float3 &operator-=(const Float3 &t) noexcept {
        *this = *this - t;
        return *this;
    }
};

struct Float4 {
    union {
        struct {
            float x, y, z, w;
        };
        struct {
            float r, g, b, a;
        };
        float e[4];
    };

    constexpr Float4(float xyzw = 0.f) noexcept : x(xyzw), y(xyzw), z(xyzw), w(xyzw) {}
    constexpr Float4(float x_, float y_, float z_, float w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}

    inline Float4 operator+(const Float4 &t) const noexcept { return { x + t.x, y + t.y, z + t.z, w + t.w }; }
    inline Float4 operator-(const Float4 &t) const noexcept { return { x - t.x, y - t.y, z - t.z, w - t.w }; }
    inline Float4 operator*(float t) const noexcept { return { x * t, y * t, z * t, w * t }; }
    inline Float4 operator/(float t) const noexcept { return *this * (1.f / t); }

    inline Float4 &operator+=(const Float4 &t) noexcept {
        *this = *this + t;
        return *this;
    }
    inline Float4 &operator-=(const Float4 &t) noexcept {
        *this = *this - t;
        return *this;
    }
};

struct Mat4 {
    union {
        struct {
            Float4 r0;
            Float4 r1;
            Float4 r2;
            Float4 r3;
        };
        float e[16];
        float re[4][4];
        DirectX::XMMATRIX dx_mat;
    };

    constexpr Mat4() noexcept : r0(0.f), r1(0.f), r2(0.f), r3(0.f) {}
    constexpr Mat4(float m00, float m01, float m02, float m03,
                   float m10, float m11, float m12, float m13,
                   float m20, float m21, float m22, float m23,
                   float m30, float m31, float m32, float m33) noexcept
        : r0(m00, m01, m02, m03), r1(m10, m11, m12, m13), r2(m20, m21, m22, m23), r3(m30, m31, m32, m33) {}

    Mat4(float e_[16]) noexcept { std::copy(e_, e_ + 16, e); }
    Mat4(DirectX::XMMATRIX m) noexcept { dx_mat = m; }
    operator DirectX::XMMATRIX() const noexcept { return dx_mat; }

    DirectX::XMMATRIX operator*(const Mat4 &t) const noexcept { return dx_mat * t.dx_mat; }

    Mat4 GetInverse() const noexcept {
        auto ret = DirectX::XMMatrixInverse(nullptr, dx_mat);
        return ret;
    }

    Mat4 GetTranspose() const noexcept { return DirectX::XMMatrixTranspose(dx_mat); }
};
}// namespace Pupil::util