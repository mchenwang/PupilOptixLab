#include "camera.h"

#include "cuda/util.h"

namespace Pupil::world {
CameraHelper::CameraHelper() noexcept {
}

CameraHelper::~CameraHelper() noexcept {
    CUDA_FREE(m_camera_cuda_memory);
}

void CameraHelper::Reset(const util::CameraDesc &desc) noexcept {
    m_camera.SetProjectionFactor(desc.fov_y, desc.aspect_ratio, desc.near_clip, desc.far_clip);
    m_camera.SetWorldTransform(desc.to_world);
    m_desc = desc;
    m_camera_cuda_memory_dirty = true;
}

void CameraHelper::SetFov(float fov) noexcept {
    m_desc.fov_y = fov;
    if (m_desc.fov_y < 0.012f)
        m_desc.fov_y = 0.012f;
    else if (m_desc.fov_y > 180.f)
        m_desc.fov_y = 180.f;
    m_camera.SetFov(m_desc.fov_y);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::SetFovDelta(float fov_delta) noexcept {
    m_desc.fov_y += fov_delta;
    // Minimum angle is 0.00001 * 2 * 180 / pi (XMMatrixPerspectiveFovRH)
    if (m_desc.fov_y < 0.012f)
        m_desc.fov_y = 0.012f;
    else if (m_desc.fov_y > 180.f)
        m_desc.fov_y = 180.f;
    m_camera.SetFov(m_desc.fov_y);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::SetAspectRatio(float aspect_ration) noexcept {
    m_desc.aspect_ratio = aspect_ration;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::SetNearClip(float near_clip) noexcept {
    m_desc.near_clip = near_clip;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::SetFarClip(float far_clip) noexcept {
    m_desc.far_clip = far_clip;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::SetWorldTransform(util::Transform to_world) noexcept {
    m_desc.to_world = to_world;
    m_camera.SetWorldTransform(to_world);
    m_camera_cuda_memory_dirty = true;
}

void CameraHelper::Rotate(float delta_x, float delta_y) noexcept {
    m_camera.Rotate(delta_x, delta_y);
    m_camera_cuda_memory_dirty = true;
}
void CameraHelper::Move(util::Float3 translation) noexcept {
    m_camera.Move(translation);
    m_camera_cuda_memory_dirty = true;
}

CUdeviceptr CameraHelper::GetCudaMemory() noexcept {
    if (m_camera_cuda_memory_dirty) {
        auto sample_to_camera = m_camera.GetSampleToCameraMatrix();
        auto camera_to_world = m_camera.GetToWorldMatrix();

        m_desc.to_world.matrix = camera_to_world;

        m_optix_camera.sample_to_camera = ToCudaType(sample_to_camera);
        m_optix_camera.camera_to_world = ToCudaType(camera_to_world);
    }

    if (m_camera_cuda_memory == 0) {
        m_camera_cuda_memory = cuda::CudaMemcpyToDevice(&m_optix_camera, sizeof(m_optix_camera));
    } else if (m_camera_cuda_memory_dirty) {
        cuda::CudaMemcpyToDevice(m_camera_cuda_memory, &m_optix_camera, sizeof(m_optix_camera));
    }

    m_camera_cuda_memory_dirty = false;

    return m_camera_cuda_memory;
}

std::tuple<util::Float3, util::Float3, util::Float3> CameraHelper::GetCameraCoordinateSystem() const noexcept {
    return m_camera.GetCameraCoordinateSystem();
}
}// namespace Pupil::world