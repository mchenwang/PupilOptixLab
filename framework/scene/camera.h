#pragma once

#include "util/transform.h"

namespace Pupil {
    struct CameraDesc {
        float fov_y;
        float aspect_ratio;
        float near_clip = 0.01f;
        float far_clip  = 10000.f;

        util::Transform to_world;
    };

    class Camera {
    public:
        static float sensitivity;
        static float sensitivity_scale;

        constexpr static util::Float3 X{1.f, 0.f, 0.f};
        constexpr static util::Float3 Y{0.f, 1.f, 0.f};
        constexpr static util::Float3 Z{0.f, 0.f, 1.f};

        Camera() noexcept;
        Camera(const CameraDesc&) noexcept;
        ~Camera() noexcept;

        util::Float3 GetPosition() const noexcept;

        util::Mat4 GetSampleToCameraMatrix() noexcept;
        util::Mat4 GetProjectionMatrix() noexcept;
        util::Mat4 GetToWorldMatrix() noexcept;
        util::Mat4 GetViewMatrix() noexcept;

        std::tuple<util::Float3, util::Float3, util::Float3> GetCameraCoordinateSystem() const noexcept;

        void SetFov(float fov) noexcept;
        void SetFovDelta(float fov_delta) noexcept;
        void SetAspectRatio(float aspect_ratio) noexcept;
        void SetNearClip(float near_clip) noexcept;
        void SetFarClip(float far_clip) noexcept;
        void SetProjectionFactor(float fov_y, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
        void SetWorldTransform(util::Transform to_world) noexcept;

        float GetFovY() noexcept;
        float GetAspectRatio() noexcept;
        float GetNearClip() noexcept;
        float GetFarClip() noexcept;

        void Rotate(float delta_x, float delta_y) noexcept;
        void Move(util::Float3 delta) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil