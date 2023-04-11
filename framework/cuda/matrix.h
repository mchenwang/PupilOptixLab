#pragma once

#include "vec_math.h"

struct mat4x4 {
    float4 r0, r1, r2, r3;
};

CUDA_INLINE CUDA_HOSTDEVICE float4 operator*(const mat4x4 &m, const float4 &v) noexcept {
    return make_float4(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}
