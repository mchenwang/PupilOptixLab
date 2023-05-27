#include "camera.h"

namespace Pupil::util {
float Camera::sensitivity = 0.05f;
float Camera::sensitivity_scale = 1.f;

Mat4 Camera::GetSampleToCameraMatrix() noexcept {
    if (m_projection_dirty) {
        auto proj = DirectX::XMMatrixPerspectiveFovRH(m_fov_y / 180.f * 3.14159265358979323846f, m_aspect_ratio, m_near_clip, m_far_clip);

        m_proj = DirectX::XMMatrixTranspose(proj);
        m_sample_to_camera = DirectX::XMMatrixTranspose(
            DirectX::XMMatrixInverse(
                nullptr, proj *
                             DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) *
                             DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f)));
        m_projection_dirty = false;
    }
    return m_sample_to_camera;
}
Mat4 Camera::GetProjectionMatrix() noexcept {
    if (m_projection_dirty) {
        auto proj = DirectX::XMMatrixPerspectiveFovRH(m_fov_y / 180.f * 3.14159265358979323846f, m_aspect_ratio, m_near_clip, m_far_clip);

        m_proj = DirectX::XMMatrixTranspose(proj);
        m_sample_to_camera = DirectX::XMMatrixTranspose(
            DirectX::XMMatrixInverse(
                nullptr, proj *
                             DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) *
                             DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f)));
        m_projection_dirty = false;
    }
    return m_proj;
}
Mat4 Camera::GetToWorldMatrix() noexcept {
    if (m_to_world_dirty) {
        m_view =
            m_rotate *
            Mat4(1.f, 0.f, 0.f, -m_position.x,
                 0.f, 1.f, 0.f, -m_position.y,
                 0.f, 0.f, 1.f, -m_position.z,
                 0.f, 0.f, 0.f, 1.f);
        m_to_world = m_view.GetInverse();
        m_to_world_dirty = false;
    }
    return m_to_world;
}
Mat4 Camera::GetViewMatrix() noexcept {
    if (m_to_world_dirty) {
        m_view =
            m_rotate *
            Mat4(1.f, 0.f, 0.f, -m_position.x,
                 0.f, 1.f, 0.f, -m_position.y,
                 0.f, 0.f, 1.f, -m_position.z,
                 0.f, 0.f, 0.f, 1.f);
        m_to_world = m_view.GetInverse();
        m_to_world_dirty = false;
    }
    return m_view;
}

std::tuple<Float3, Float3, Float3> Camera::GetCameraCoordinateSystem() const noexcept {
    auto right = Transform::TransformVector(X, m_rotate_inv);
    auto up = Transform::TransformVector(Y, m_rotate_inv);
    auto forward = Transform::TransformVector(Z, m_rotate_inv);
    return { right, up, forward };
}

void Camera::SetProjectionFactor(float fov_y, float aspect_ratio, float near_clip, float far_clip) noexcept {
    m_fov_y = fov_y;
    m_aspect_ratio = aspect_ratio;
    m_near_clip = near_clip;
    m_far_clip = far_clip;
    m_projection_dirty = true;
}
void Camera::SetFov(float fov) noexcept {
    m_fov_y = fov;
    m_projection_dirty = true;
}
void Camera::SetWorldTransform(Transform to_world) noexcept {
    m_to_world = to_world.matrix;
    m_position.x = m_to_world.r0.w;
    m_position.y = m_to_world.r1.w;
    m_position.z = m_to_world.r2.w;

    m_rotate = to_world.matrix.GetTranspose();
    m_rotate.r3.x = 0.f;
    m_rotate.r3.y = 0.f;
    m_rotate.r3.z = 0.f;

    m_rotate_inv = m_rotate.GetTranspose();

    m_view =
        m_rotate *
        Mat4(1.f, 0.f, 0.f, -m_position.x,
             0.f, 1.f, 0.f, -m_position.y,
             0.f, 0.f, 1.f, -m_position.z,
             0.f, 0.f, 0.f, 1.f);

    m_to_world_dirty = false;
}

void Camera::Rotate(float delta_x, float delta_y) noexcept {
    Transform pitch;
    pitch.Rotate(X.x, X.y, X.z, delta_y);
    Transform yaw;
    yaw.Rotate(Y.x, Y.y, Y.z, delta_x);

    m_rotate = pitch.matrix * m_rotate * yaw.matrix;
    m_rotate_inv = m_rotate.GetTranspose();
    m_to_world_dirty = true;
}

void Camera::Move(Float3 delta) noexcept {
    delta = Transform::TransformVector(delta, m_rotate_inv);
    Transform translation;
    translation.Translate(delta.x, delta.y, delta.z);
    m_position = Transform::TransformPoint(m_position, translation.matrix);
    m_to_world_dirty = true;
}
}// namespace Pupil::util
