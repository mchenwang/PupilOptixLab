#include "camera.h"

#include <cuda.h>

namespace Pupil {
    float Camera::sensitivity       = 0.05f;
    float Camera::sensitivity_scale = 1.f;

    static const Angle fov_min(Angle::DegreeToRadian(0.012f));
    static const Angle fov_max(Angle::DegreeToRadian(180.f));

    struct Camera::Impl {
        Angle fov_y;
        float aspect_ratio;
        float near_clip;
        float far_clip;

        Float3 position;

        Matrix4x4f rotate;
        Matrix4x4f rotate_inv;

        bool       to_world_dirty = true;
        Matrix4x4f to_world;// camera to world
        Matrix4x4f view;    // world to camera

        bool       projection_dirty = true;
        Matrix4x4f sample_to_camera;// screen to camera
        Matrix4x4f proj;            // camera to screen
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

    Float3 Camera::GetPosition() const noexcept { return m_impl->position; }
    Angle  Camera::GetFovY() noexcept { return m_impl->fov_y; }
    float  Camera::GetAspectRatio() noexcept { return m_impl->aspect_ratio; }
    float  Camera::GetNearClip() noexcept { return m_impl->near_clip; }
    float  Camera::GetFarClip() noexcept { return m_impl->far_clip; }

    Matrix4x4f Camera::GetSampleToCameraMatrix() noexcept {
        if (m_impl->projection_dirty) {
            m_impl->proj = Pupil::MakePerspectiveMatrixRH(
                m_impl->fov_y.GetRadian(), m_impl->aspect_ratio, m_impl->near_clip, m_impl->far_clip);
            m_impl->sample_to_camera = Pupil::Inverse(
                Pupil::MakeScaling(0.5f, 0.5f, 1.f) *
                Pupil::MakeTranslation(1.f, 1.f, 0.f) *
                m_impl->proj);

            m_impl->projection_dirty = false;
        }
        return m_impl->sample_to_camera;
    }

    Matrix4x4f Camera::GetProjectionMatrix() noexcept {
        if (m_impl->projection_dirty) {
            m_impl->proj = Pupil::MakePerspectiveMatrixRH(
                m_impl->fov_y.GetRadian(), m_impl->aspect_ratio, m_impl->near_clip, m_impl->far_clip);
            m_impl->sample_to_camera = Pupil::Inverse(
                Pupil::MakeScaling(0.5f, 0.5f, 1.f) *
                Pupil::MakeTranslation(1.f, 1.f, 0.f) *
                m_impl->proj);

            m_impl->projection_dirty = false;
        }
        return m_impl->proj;
    }

    Matrix4x4f Camera::GetToWorldMatrix() noexcept {
        if (m_impl->to_world_dirty) {
            m_impl->view = m_impl->rotate *
                           Pupil::MakeTranslation(-m_impl->position.x, -m_impl->position.y, -m_impl->position.z);
            m_impl->to_world       = Pupil::Inverse(m_impl->view);
            m_impl->to_world_dirty = false;
        }
        return m_impl->to_world;
    }

    Matrix4x4f Camera::GetViewMatrix() noexcept {
        if (m_impl->to_world_dirty) {
            m_impl->view = m_impl->rotate *
                           Pupil::MakeTranslation(-m_impl->position.x, -m_impl->position.y, -m_impl->position.z);
            m_impl->to_world       = Pupil::Inverse(m_impl->view);
            m_impl->to_world_dirty = false;
        }
        return m_impl->view;
    }

    std::tuple<Float3, Float3, Float3> Camera::GetCameraCoordinateSystem() const noexcept {
        auto right   = Float3(m_impl->rotate_inv * Vector4f(X, 0.f));
        auto up      = Float3(m_impl->rotate_inv * Vector4f(Y, 0.f));
        auto forward = Float3(m_impl->rotate_inv * Vector4f(Z, 0.f));
        return {right, up, forward};
    }

    void Camera::SetProjectionFactor(Angle fov_y, float aspect_ratio, float near_clip, float far_clip) noexcept {
        if (fov_y.GetRadian() < fov_min.GetRadian())
            fov_y = fov_min;
        else if (fov_y.GetRadian() > fov_max.GetRadian())
            fov_y = fov_max;
        m_impl->fov_y            = fov_y;
        m_impl->aspect_ratio     = aspect_ratio;
        m_impl->near_clip        = near_clip;
        m_impl->far_clip         = far_clip;
        m_impl->projection_dirty = true;
    }

    void Camera::SetFov(float fov) noexcept {
        SetFov(Angle::MakeFromDegree(fov));
    }

    void Camera::SetFovDelta(float fov_delta) noexcept {
        SetFov(Angle::MakeFromDegree(fov_delta) + m_impl->fov_y);
    }

    void Camera::SetFov(Angle fov) noexcept {
        if (fov.GetRadian() < fov_min.GetRadian())
            fov = fov_min;
        else if (fov.GetRadian() > fov_max.GetRadian())
            fov = fov_max;
        m_impl->fov_y            = fov;
        m_impl->projection_dirty = true;
    }

    void Camera::SetFovDelta(Angle fov_delta) noexcept {
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

        m_impl->rotate      = Pupil::Transpose(to_world.matrix);
        m_impl->rotate.r3.x = 0.f;
        m_impl->rotate.r3.y = 0.f;
        m_impl->rotate.r3.z = 0.f;

        m_impl->rotate_inv = Pupil::Transpose(m_impl->rotate);

        m_impl->view = m_impl->rotate *
                       Pupil::MakeTranslation(-m_impl->position.x, -m_impl->position.y, -m_impl->position.z);

        m_impl->to_world_dirty = false;
    }

    void Camera::Rotate(float delta_x, float delta_y) noexcept {
        Quaternion pitch(X, Angle::MakeFromDegree(delta_x));
        Quaternion yaw(Y, Angle::MakeFromDegree(delta_x));

        m_impl->rotate =
            Pupil::FillDiagonal4x4(pitch.GetRotation(), 1.f) *
            m_impl->rotate *
            Pupil::FillDiagonal4x4(yaw.GetRotation(), 1.f);

        m_impl->rotate_inv     = Pupil::Transpose(m_impl->rotate);
        m_impl->to_world_dirty = true;
    }

    void Camera::Move(Float3 delta) noexcept {
        delta = Vector3f(m_impl->rotate_inv * Vector4f(delta, 0.f));

        auto translation       = Pupil::MakeTranslation(delta.x, delta.y, delta.z);
        m_impl->position       = Vector3f(translation * Vector4f(m_impl->position, 1.f));
        m_impl->to_world_dirty = true;
    }
}// namespace Pupil
