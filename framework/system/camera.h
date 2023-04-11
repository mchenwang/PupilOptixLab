#pragma once

#include "util/camera.h"
#include "optix/scene/camera.h"

#include <cuda.h>

namespace Pupil {
class CameraHelper {
private:
    bool m_dirty;
    util::CameraDesc m_desc;
    Pupil::util::Camera m_camera;
    Pupil::optix::Camera m_optix_camera;

    CUdeviceptr m_camera_cuda_memory = 0;

public:
    CameraHelper(const util::CameraDesc &desc) noexcept;
    ~CameraHelper() noexcept;

    void Reset(const util::CameraDesc &desc) noexcept;

    void SetFov(float fov) noexcept;
    void SetFovDelta(float fov_delta) noexcept;
    void SetAspectRatio(float aspect_ration) noexcept;
    void SetNearClip(float near_clip) noexcept;
    void SetFarClip(float far_clip) noexcept;
    void SetWorldTransform(util::Transform to_world) noexcept;

    void Rotate(float delta_x, float delta_y) noexcept;
    void Move(util::Float3 translation) noexcept;

    CUdeviceptr GetCudaMemory() noexcept;

    Pupil::util::Camera &GetUtilCamera() noexcept { return m_camera; }
    Pupil::optix::Camera &GetOptixCamera() noexcept { return m_optix_camera; }

    std::tuple<util::Float3, util::Float3, util::Float3> GetCameraCoordinateSystem() const noexcept;
};
}// namespace Pupil