#pragma once

#include "type.h"
#include "transform.h"

namespace util {
class Camera {
private:
    float m_fov_y;
    float m_aspect_ratio;
    float m_near_clip;
    float m_far_clip;

    Float3 m_right;
    Float3 m_up;
    Float3 m_forward;
    Float3 m_position;

    bool m_to_world_dirty = true;
    bool m_projection_dirty = true;
    Mat4 m_to_world;
    Mat4 m_sample_to_camera;

public:
    Camera() noexcept = default;

    Mat4 GetSampleToCameraMatrix() noexcept {
        if (m_projection_dirty) {
            m_sample_to_camera = DirectX::XMMatrixTranspose(
                DirectX::XMMatrixInverse(
                    nullptr,
                    DirectX::XMMatrixPerspectiveFovRH(m_fov_y / 180.f * 3.14159265358979323846f, m_aspect_ratio, m_near_clip, m_far_clip) *
                        DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) *
                        DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f)));
            m_projection_dirty = false;
        }
        return m_sample_to_camera;
    }

    Mat4 GetToWorldMatrix() noexcept {
        if (m_to_world_dirty) {
            Mat4 view_matrix =
                Mat4(m_right.x, m_up.x, m_forward.x, 0.f,
                     m_right.y, m_up.y, m_forward.y, 0.f,
                     m_right.z, m_up.z, m_forward.z, 0.f,
                     0.f, 0.f, 0.f, 1.f) *
                Mat4(1.f, 0.f, 0.f, -m_position.x,
                     0.f, 1.f, 0.f, -m_position.y,
                     0.f, 0.f, 1.f, -m_position.z,
                     0.f, 0.f, 0.f, 1.f);
            m_to_world = view_matrix.GetInverse();
            m_to_world_dirty = false;
        }
        return m_to_world;
    }

    void SetProjectionFactor(float fov_y, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept {
        m_fov_y = fov_y;
        m_aspect_ratio = aspect_ratio;
        m_near_clip = near_clip;
        m_far_clip = far_clip;
        m_projection_dirty = true;
    }

    void UpdateFov(float fov) noexcept {
        m_fov_y = fov;
        m_projection_dirty = true;
    }

    void SetWorldTransform(Transform to_world) noexcept {
        m_to_world = to_world.matrix;
        m_position.x = m_to_world.r0.w;
        m_position.y = m_to_world.r1.w;
        m_position.z = m_to_world.r2.w;

        Mat4 view_matrix = m_to_world.GetInverse();

        m_right.x = view_matrix.r0.x;
        m_right.y = view_matrix.r1.x;
        m_right.z = view_matrix.r2.x;

        m_up.x = view_matrix.r0.y;
        m_up.y = view_matrix.r1.y;
        m_up.z = view_matrix.r2.y;

        m_forward.x = view_matrix.r0.z;
        m_forward.y = view_matrix.r1.z;
        m_forward.z = view_matrix.r2.z;

        m_to_world_dirty = false;
    }

    void Pitch(float angle) noexcept {
        Transform rotate;
        rotate.Rotate(m_right.x, m_right.y, m_right.z, angle);
        m_up = Transform::TransformVector(m_up, rotate.matrix);
        m_forward = Transform::TransformVector(m_forward, rotate.matrix);
        m_to_world_dirty = true;
    }

    void Yaw(float angle) noexcept {
        Transform rotate;
        rotate.Rotate(m_up.x, m_up.y, m_up.z, angle);
        m_right = Transform::TransformVector(m_right, rotate.matrix);
        m_forward = Transform::TransformVector(m_forward, rotate.matrix);
        m_to_world_dirty = true;
    }

    void Roll(float angle) noexcept {
        Transform rotate;
        rotate.Rotate(m_forward.x, m_forward.y, m_forward.z, angle);
        m_right = Transform::TransformVector(m_right, rotate.matrix);
        m_up = Transform::TransformVector(m_up, rotate.matrix);
        m_to_world_dirty = true;
    }

    void RotateY(float angle) noexcept {
        Transform rotate;
        rotate.Rotate(0.f, 1.f, 0.f, angle);
        m_up = Transform::TransformVector(m_up, rotate.matrix);
        m_right = Transform::TransformVector(m_right, rotate.matrix);
        m_forward = Transform::TransformVector(m_forward, rotate.matrix);
        m_to_world_dirty = true;
    }

    void Move(Float3 translation) {
        Transform translate;
        translate.Translate(translation.x, translation.y, translation.z);
        m_position = Transform::TransformPoint(m_position, translate.matrix);
        m_to_world_dirty = true;
    }
};
}// namespace util