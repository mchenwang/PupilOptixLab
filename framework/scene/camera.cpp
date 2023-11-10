#include "camera.h"

#include <cuda.h>

namespace Pupil {
    using namespace util;
    float Camera::sensitivity       = 0.05f;
    float Camera::sensitivity_scale = 1.f;

    struct Camera::Impl {
        float fov_y;
        float aspect_ratio;
        float near_clip;
        float far_clip;

        util::Float3 position;

        util::Mat4 rotate;
        util::Mat4 rotate_inv;

        bool       to_world_dirty = true;
        util::Mat4 to_world;// camera to world
        util::Mat4 view;    // world to camera

        bool       projection_dirty = true;
        util::Mat4 sample_to_camera;// screen to camera
        util::Mat4 proj;            // camera to screen
    };

    Camera::Camera() noexcept {
        m_impl = new Impl();
    }

    Camera::Camera(const CameraDesc& desc) noexcept {
        SetProjectionFactor(desc.fov_y, desc.aspect_ratio, desc.near_clip, desc.far_clip);
        SetWorldTransform(desc.to_world);
    }

    Camera::~Camera() noexcept {
        delete m_impl;
    }

    util::Float3 Camera::GetPosition() const noexcept { return m_impl->position; }
    float        Camera::GetFovY() noexcept { return m_impl->fov_y; }
    float        Camera::GetAspectRatio() noexcept { return m_impl->aspect_ratio; }
    float        Camera::GetNearClip() noexcept { return m_impl->near_clip; }
    float        Camera::GetFarClip() noexcept { return m_impl->far_clip; }

    Mat4 Camera::GetSampleToCameraMatrix() noexcept {
        if (m_impl->projection_dirty) {
            auto proj = DirectX::XMMatrixPerspectiveFovRH(m_impl->fov_y / 180.f * 3.14159265358979323846f, m_impl->aspect_ratio, m_impl->near_clip, m_impl->far_clip);

            m_impl->proj             = DirectX::XMMatrixTranspose(proj);
            m_impl->sample_to_camera = DirectX::XMMatrixTranspose(
                DirectX::XMMatrixInverse(
                    nullptr, proj * DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) * DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f)));
            m_impl->projection_dirty = false;
        }
        return m_impl->sample_to_camera;
    }

    Mat4 Camera::GetProjectionMatrix() noexcept {
        if (m_impl->projection_dirty) {
            auto proj = DirectX::XMMatrixPerspectiveFovRH(m_impl->fov_y / 180.f * 3.14159265358979323846f, m_impl->aspect_ratio, m_impl->near_clip, m_impl->far_clip);

            m_impl->proj             = DirectX::XMMatrixTranspose(proj);
            m_impl->sample_to_camera = DirectX::XMMatrixTranspose(
                DirectX::XMMatrixInverse(
                    nullptr, proj * DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) * DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f)));
            m_impl->projection_dirty = false;
        }
        return m_impl->proj;
    }

    Mat4 Camera::GetToWorldMatrix() noexcept {
        if (m_impl->to_world_dirty) {
            m_impl->view =
                m_impl->rotate *
                Mat4(1.f, 0.f, 0.f, -m_impl->position.x, 0.f, 1.f, 0.f, -m_impl->position.y, 0.f, 0.f, 1.f, -m_impl->position.z, 0.f, 0.f, 0.f, 1.f);
            m_impl->to_world       = m_impl->view.GetInverse();
            m_impl->to_world_dirty = false;
        }
        return m_impl->to_world;
    }

    Mat4 Camera::GetViewMatrix() noexcept {
        if (m_impl->to_world_dirty) {
            m_impl->view =
                m_impl->rotate *
                Mat4(1.f, 0.f, 0.f, -m_impl->position.x, 0.f, 1.f, 0.f, -m_impl->position.y, 0.f, 0.f, 1.f, -m_impl->position.z, 0.f, 0.f, 0.f, 1.f);
            m_impl->to_world       = m_impl->view.GetInverse();
            m_impl->to_world_dirty = false;
        }
        return m_impl->view;
    }

    std::tuple<Float3, Float3, Float3> Camera::GetCameraCoordinateSystem() const noexcept {
        auto right   = Transform::TransformVector(X, m_impl->rotate_inv);
        auto up      = Transform::TransformVector(Y, m_impl->rotate_inv);
        auto forward = Transform::TransformVector(Z, m_impl->rotate_inv);
        return {right, up, forward};
    }

    void Camera::SetProjectionFactor(float fov_y, float aspect_ratio, float near_clip, float far_clip) noexcept {
        if (fov_y < 0.012f)
            fov_y = 0.012f;
        else if (fov_y > 180.f)
            fov_y = 180.f;
        m_impl->fov_y            = fov_y;
        m_impl->aspect_ratio     = aspect_ratio;
        m_impl->near_clip        = near_clip;
        m_impl->far_clip         = far_clip;
        m_impl->projection_dirty = true;
    }

    void Camera::SetFov(float fov) noexcept {
        // Minimum angle is 0.00001 * 2 * 180 / pi (XMMatrixPerspectiveFovRH)
        if (fov < 0.012f)
            fov = 0.012f;
        else if (fov > 180.f)
            fov = 180.f;
        m_impl->fov_y            = fov;
        m_impl->projection_dirty = true;
    }

    void Camera::SetFovDelta(float fov_delta) noexcept {
        SetFov(fov_delta + m_impl->fov_y);
    }

    void Camera::SetAspectRatio(float aspect_ratio) noexcept {
        m_impl->aspect_ratio     = aspect_ratio;
        m_impl->projection_dirty = true;
    }

    void Camera::SetNearClip(float near_clip) noexcept {
        m_impl->near_clip        = near_clip;
        m_impl->projection_dirty = true;
    }

    void Camera::SetFarClip(float far_clip) noexcept {
        m_impl->far_clip         = far_clip;
        m_impl->projection_dirty = true;
    }

    void Camera::SetWorldTransform(Transform to_world) noexcept {
        m_impl->to_world   = to_world.matrix;
        m_impl->position.x = m_impl->to_world.r0.w;
        m_impl->position.y = m_impl->to_world.r1.w;
        m_impl->position.z = m_impl->to_world.r2.w;

        m_impl->rotate      = to_world.matrix.GetTranspose();
        m_impl->rotate.r3.x = 0.f;
        m_impl->rotate.r3.y = 0.f;
        m_impl->rotate.r3.z = 0.f;

        m_impl->rotate_inv = m_impl->rotate.GetTranspose();

        m_impl->view =
            m_impl->rotate *
            Mat4(1.f, 0.f, 0.f, -m_impl->position.x, 0.f, 1.f, 0.f, -m_impl->position.y, 0.f, 0.f, 1.f, -m_impl->position.z, 0.f, 0.f, 0.f, 1.f);

        m_impl->to_world_dirty = false;
    }

    void Camera::Rotate(float delta_x, float delta_y) noexcept {
        Transform pitch;
        pitch.Rotate(X.x, X.y, X.z, delta_y);
        Transform yaw;
        yaw.Rotate(Y.x, Y.y, Y.z, delta_x);

        m_impl->rotate         = pitch.matrix * m_impl->rotate * yaw.matrix;
        m_impl->rotate_inv     = m_impl->rotate.GetTranspose();
        m_impl->to_world_dirty = true;
    }

    void Camera::Move(Float3 delta) noexcept {
        delta = Transform::TransformVector(delta, m_impl->rotate_inv);
        Transform translation;
        translation.Translate(delta.x, delta.y, delta.z);
        m_impl->position       = Transform::TransformPoint(m_impl->position, translation.matrix);
        m_impl->to_world_dirty = true;
    }
}// namespace Pupil
