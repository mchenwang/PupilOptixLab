#include "camera.h"
#include "cuda_util/util.h"

namespace optix_util {

CameraHelper::CameraHelper(const CameraDesc &desc) noexcept {
    m_camera.SetProjectionFactor(desc.fov_y, desc.aspect_ratio, desc.near_clip, desc.far_clip);
    m_camera.SetWorldTransform(desc.to_world);
    m_desc = desc;
    m_dirty = true;
}

CameraHelper::~CameraHelper() noexcept {
    CUDA_FREE(m_camera_cuda_memory);
}

void CameraHelper::Reset(const CameraDesc &desc) noexcept {
    m_camera.SetProjectionFactor(desc.fov_y, desc.aspect_ratio, desc.near_clip, desc.far_clip);
    m_camera.SetWorldTransform(desc.to_world);
    m_desc = desc;
    m_dirty = true;
}

void CameraHelper::SetFov(float fov) noexcept {
    m_camera.SetFov(fov);
    m_desc.fov_y = fov;
    m_dirty = true;
}
void CameraHelper::SetFovDelta(float fov_delta) noexcept {
    m_desc.fov_y += fov_delta;
    // Minimum angle is 0.00001 * 2 * 180 / pi (XMMatrixPerspectiveFovRH)
    if (m_desc.fov_y < 0.012f)
        m_desc.fov_y = 0.012f;
    else if (m_desc.fov_y > 180.f)
        m_desc.fov_y = 180.f;
    m_camera.SetFov(m_desc.fov_y);
    m_dirty = true;
}
void CameraHelper::SetAspectRatio(float aspect_ration) noexcept {
    m_desc.aspect_ratio = aspect_ration;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_dirty = true;
}
void CameraHelper::SetNearClip(float near_clip) noexcept {
    m_desc.near_clip = near_clip;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_dirty = true;
}
void CameraHelper::SetFarClip(float far_clip) noexcept {
    m_desc.far_clip = far_clip;
    m_camera.SetProjectionFactor(
        m_desc.fov_y, m_desc.aspect_ratio, m_desc.near_clip, m_desc.far_clip);
    m_dirty = true;
}
void CameraHelper::SetWorldTransform(util::Transform to_world) noexcept {
    m_desc.to_world = to_world;
    m_camera.SetWorldTransform(to_world);
    m_dirty = true;
}

void CameraHelper::Rotate(float delta_x, float delta_y) noexcept {
    m_camera.Rotate(delta_x, delta_y);
    m_dirty = true;
}
void CameraHelper::Move(util::Float3 translation) noexcept {
    m_camera.Move(translation);
    m_dirty = true;
}

CUdeviceptr CameraHelper::GetCudaMemory() noexcept {
    if (m_dirty) {
        auto sample_to_camera = m_camera.GetSampleToCameraMatrix();
        auto camera_to_world = m_camera.GetToWorldMatrix();

        m_optix_camera.sample_to_camera.r0 = make_float4(sample_to_camera.r0.x, sample_to_camera.r0.y, sample_to_camera.r0.z, sample_to_camera.r0.w);
        m_optix_camera.sample_to_camera.r1 = make_float4(sample_to_camera.r1.x, sample_to_camera.r1.y, sample_to_camera.r1.z, sample_to_camera.r1.w);
        m_optix_camera.sample_to_camera.r2 = make_float4(sample_to_camera.r2.x, sample_to_camera.r2.y, sample_to_camera.r2.z, sample_to_camera.r2.w);
        m_optix_camera.sample_to_camera.r3 = make_float4(sample_to_camera.r3.x, sample_to_camera.r3.y, sample_to_camera.r3.z, sample_to_camera.r3.w);

        m_optix_camera.camera_to_world.r0 = make_float4(-camera_to_world.r0.x, camera_to_world.r0.y, -camera_to_world.r0.z, camera_to_world.r0.w);
        m_optix_camera.camera_to_world.r1 = make_float4(-camera_to_world.r1.x, camera_to_world.r1.y, -camera_to_world.r1.z, camera_to_world.r1.w);
        m_optix_camera.camera_to_world.r2 = make_float4(-camera_to_world.r2.x, camera_to_world.r2.y, -camera_to_world.r2.z, camera_to_world.r2.w);
        m_optix_camera.camera_to_world.r3 = make_float4(camera_to_world.r3.x, camera_to_world.r3.y, camera_to_world.r3.z, camera_to_world.r3.w);
    }

    if (m_camera_cuda_memory == 0) {
        m_camera_cuda_memory = cuda::CudaMemcpyToDevice(&m_optix_camera, sizeof(m_optix_camera));
    } else if (m_dirty) {
        cuda::CudaMemcpyToDevice(m_camera_cuda_memory, &m_optix_camera, sizeof(m_optix_camera));
    }

    m_dirty = false;

    return m_camera_cuda_memory;
}

std::tuple<util::Float3, util::Float3, util::Float3> CameraHelper::GetCameraCoordinateSystem() const noexcept {
    return m_camera.GetCameraCoordinateSystem();
}

}// namespace optix_util