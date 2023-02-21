#pragma once

#include "cuda_util/preprocessor.h"
#include "common/camera.h"

#include <vector_types.h>
#include <cuda.h>

namespace optix_util {
struct Camera {
    struct {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 r3;
    } sample_to_camera, camera_to_world;

    // #ifdef PUPIL_OPTIX_LAUNCHER_SIDE
    //     void SetCameraTransform(float fov, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
    //     void SetWorldTransform(float matrix[16]) noexcept;
    // #endif
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
struct CameraDesc {
    float fov_y;
    float aspect_ratio;
    float near_clip = 0.01f;
    float far_clip = 10000.f;

    util::Transform to_world;
};

class CameraHelper {
private:
    bool m_dirty;
    CameraDesc m_desc;
    util::Camera m_camera;
    optix_util::Camera m_optix_camera;

    CUdeviceptr m_camera_cuda_memory = 0;

public:
    CameraHelper(const CameraDesc &desc) noexcept;
    ~CameraHelper() noexcept;

    void SetFov(float fov) noexcept;
    void SetAspectRatio(float aspect_ration) noexcept;
    void SetNearClip(float near_clip) noexcept;
    void SetFarClip(float far_clip) noexcept;
    void SetWorldTransform(util::Transform to_world) noexcept;

    void Pitch(float angle) noexcept;
    void Yaw(float angle) noexcept;
    void Roll(float angle) noexcept;
    void RotateY(float angle) noexcept;
    void Move(util::Float3 translation) noexcept;

    CUdeviceptr GetCudaMemory() noexcept;
};
#endif
}// namespace optix_util