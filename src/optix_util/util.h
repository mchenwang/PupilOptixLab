#pragma once

#include <optix.h>
#include "cuda_util/preprocessor.h"
#include "cuda_util/vec_math.h"

namespace optix_util {
CUDA_DEVICE static const float EPS = 0.000001f;

CUDA_INLINE CUDA_DEVICE void PackPointer(void *target, uint32_t &u0, uint32_t &u1) {
    const uint64_t ptr = reinterpret_cast<uint64_t>(target);
    u0 = ptr >> 32;
    u1 = ptr & 0x00000000ffffffff;
}

CUDA_INLINE CUDA_DEVICE void *UnpackPointer(uint32_t u0, uint32_t u1) {
    const uint64_t ptr = static_cast<uint64_t>(u0) << 32 | u1;
    return reinterpret_cast<void *>(ptr);
}

#ifndef PUPIL_OPTIX_LAUNCHER_SIDE
template<typename T>
CUDA_INLINE CUDA_DEVICE T *GetPRD() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(UnpackPointer(u0, u1));
}
#endif// PUPIL_OPTIX_LAUNCHER_SIDE

// https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
// return interpolation factor for triangle vertex
CUDA_INLINE CUDA_HOSTDEVICE float3 UniformSampleTriangle(const float u1, const float u2) {
    const float sqrt_u1 = sqrtf(u1);
    return make_float3(1.f - sqrt_u1, sqrt_u1 * (1.f - u2), u2 * sqrt_u1);
}

CUDA_INLINE CUDA_HOSTDEVICE float3 UniformSampleSphere(const float u1, const float u2) {
    const float z = 1.f - 2.f * u1;
    const float sin_theta = sqrtf(max(0.f, 1.f - z * z));
    const float phi = 2.f * M_PIf * u2;
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float v) {
    return abs(v) < EPS;
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float2 v) {
    return abs(v.x) < EPS && abs(v.y) < EPS;
}

CUDA_INLINE CUDA_DEVICE bool IsZero(float3 v) {
    return abs(v.x) < EPS && abs(v.y) < EPS && abs(v.z) < EPS;
}
}// namespace optix_util