#pragma once

#include "util/math.h"

namespace Pupil {
    struct CameraDesc {
        Angle fov_y;
        float aspect_ratio;
        float near_clip = 0.01f;
        float far_clip  = 10000.f;

        Transform to_world;
    };

    class Camera {
    public:
        static float sensitivity;
        static float sensitivity_scale;

        constexpr static Float3 X{1.f, 0.f, 0.f};
        constexpr static Float3 Y{0.f, 1.f, 0.f};
        constexpr static Float3 Z{0.f, 0.f, 1.f};

        Camera() noexcept;
        Camera(const CameraDesc&) noexcept;
        ~Camera() noexcept;

        Float3 GetPosition() const noexcept;

        Matrix4x4f GetSampleToCameraMatrix() noexcept;
        Matrix4x4f GetProjectionMatrix() noexcept;
        Matrix4x4f GetToWorldMatrix() noexcept;
        Matrix4x4f GetViewMatrix() noexcept;

        std::tuple<Float3, Float3, Float3> GetCameraCoordinateSystem() const noexcept;

        void SetFov(float fov) noexcept;
        void SetFovDelta(float fov_delta) noexcept;
        void SetFov(Angle fov) noexcept;
        void SetFovDelta(Angle fov_delta) noexcept;
        void SetAspectRatio(float aspect_ratio) noexcept;
        void SetNearClip(float near_clip) noexcept;
        void SetFarClip(float far_clip) noexcept;
        void SetProjectionFactor(Angle fov_y, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
        void SetWorldTransform(Transform to_world) noexcept;

        Angle GetFovY() noexcept;
        float GetAspectRatio() noexcept;
        float GetNearClip() noexcept;
        float GetFarClip() noexcept;

        void Rotate(float delta_x, float delta_y) noexcept;
        void Move(Float3 delta) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil