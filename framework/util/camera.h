#pragma once

#include "type.h"
#include "transform.h"

namespace Pupil::util {
class Camera {
private:
    float m_fov_y;
    float m_aspect_ratio;
    float m_near_clip;
    float m_far_clip;

    Float3 m_position;

    Mat4 m_rotate;
    Mat4 m_rotate_inv;

    bool m_to_world_dirty = true;
    bool m_projection_dirty = true;
    Mat4 m_to_world;
    Mat4 m_sample_to_camera;

public:
    static float sensitivity;
    static float sensitivity_scale;

    constexpr static Float3 X{ 1.f, 0.f, 0.f };
    constexpr static Float3 Y{ 0.f, 1.f, 0.f };
    constexpr static Float3 Z{ 0.f, 0.f, 1.f };

    Camera() noexcept = default;

    Mat4 GetSampleToCameraMatrix() noexcept;
    Mat4 GetToWorldMatrix() noexcept;

    std::tuple<Float3, Float3, Float3> GetCameraCoordinateSystem() const noexcept;

    void SetProjectionFactor(float fov_y, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
    void SetFov(float fov) noexcept;
    void SetWorldTransform(Transform to_world) noexcept;

    void Rotate(float delta_x, float delta_y) noexcept;
    void Move(Float3 delta) noexcept;
};
}// namespace Pupil::util