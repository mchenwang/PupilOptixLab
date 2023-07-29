#pragma once

#include "util/camera.h"
#include "render/camera.h"
#include "system/type.h"

#include <cuda.h>

namespace Pupil::world {
class CameraHelper {
private:
    util::CameraDesc m_desc;
    Pupil::util::Camera m_camera;
    Pupil::optix::Camera m_optix_camera;

    bool m_camera_cuda_memory_dirty;
    CUdeviceptr m_camera_cuda_memory = 0;

public:
    CameraHelper() noexcept;
    ~CameraHelper() noexcept;

    void Reset(const util::CameraDesc &desc) noexcept;
    util::CameraDesc GetDesc() const noexcept { return m_desc; }

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

    util::Mat4 GetSampleToCameraMatrix() noexcept { return m_camera.GetSampleToCameraMatrix(); }
    util::Mat4 GetProjectionMatrix() noexcept { return m_camera.GetProjectionMatrix(); }
    util::Mat4 GetToWorldMatrix() noexcept { return m_camera.GetToWorldMatrix(); }
    util::Mat4 GetViewMatrix() noexcept { return m_camera.GetViewMatrix(); }

    mat4x4 GetSampleToCameraCudaMatrix() noexcept { return ToCudaType(GetSampleToCameraMatrix()); }
    mat4x4 GetProjectionCudaMatrix() noexcept { return ToCudaType(GetProjectionMatrix()); }
    mat4x4 GetToWorldCudaMatrix() noexcept { return ToCudaType(GetToWorldMatrix()); }
    mat4x4 GetViewCudaMatrix() noexcept { return ToCudaType(GetViewMatrix()); }

    std::tuple<util::Float3, util::Float3, util::Float3> GetCameraCoordinateSystem() const noexcept;
};
}// namespace Pupil::world