#pragma once

#include "util/type.h"
#include "cuda/vec_math.h"
#include "cuda/matrix.h"

namespace Pupil {
inline float3 ToCudaType(const util::Float3 &v) noexcept { return make_float3(v.x, v.y, v.z); }
inline float4 ToCudaType(const util::Float4 &v) noexcept { return make_float4(v.x, v.y, v.z, v.w); }
inline mat4x4 ToCudaType(const util::Mat4 &m) noexcept {
    mat4x4 c_m;
    c_m.r0 = ToCudaType(m.r0);
    c_m.r1 = ToCudaType(m.r1);
    c_m.r2 = ToCudaType(m.r2);
    c_m.r3 = ToCudaType(m.r3);
    return c_m;
}

inline util::Float3 ToUtilType(const float3 &v) noexcept { return util::Float3(v.x, v.y, v.z); }
inline util::Float4 ToUtilType(const float4 &v) noexcept { return util::Float4(v.x, v.y, v.z, v.w); }
inline util::Mat4 ToUtilType(const mat4x4 &c_m) noexcept {
    util::Mat4 m;
    m.r0 = ToUtilType(c_m.r0);
    m.r1 = ToUtilType(c_m.r1);
    m.r2 = ToUtilType(c_m.r2);
    m.r3 = ToUtilType(c_m.r3);
    return m;
}

}// namespace Pupil