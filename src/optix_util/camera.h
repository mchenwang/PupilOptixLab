#pragma once

#include "cuda_util/preprocessor.h"

#include <vector_types.h>

namespace optix_util {
struct Camera {
    struct {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 r3;
    } sample_to_camera, camera_to_world;

#if !defined(__CUDACC__) && !defined(__CUDABE__)
    void SetCameraTransform(float fov, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
    void SetWorldTransform(float matrix[16]) noexcept;
#endif
};
}// namespace optix_util