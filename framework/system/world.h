#pragma once
#include "scene/scene.h"

#include "util/util.h"

#include <filesystem>
#include <memory>

namespace Pupil {
    // enum class EWorldEvent {
    //     CameraChange,
    //     CameraMove,
    //     CameraFovChange,
    //     CameraViewChange,
    //     RenderInstanceTransform,
    //     RenderInstanceUpdate,
    //     RenderInstanceRemove
    // };

    namespace Event {
        constexpr const char* SceneReset     = "Scene Reset";
        constexpr const char* CameraChange   = "Camera Change";
        constexpr const char* InstanceChange = "Instance Change";
    }// namespace Event

    class World : public util::Singleton<World> {
    public:
        void Init() noexcept;
        void Destroy() noexcept;

        bool LoadScene(std::filesystem::path) noexcept;
        void Upload() noexcept;

        Scene* GetScene() noexcept;

        void SetCameraFov(float fov) noexcept;
        void SetCameraFovDelta(float fov_delta) noexcept;
        void SetCameraFov(Angle fov) noexcept;
        void SetCameraFovDelta(Angle fov_delta) noexcept;
        void SetCameraAspectRatio(float aspect_ratio) noexcept;
        void SetCameraNearClip(float near_clip) noexcept;
        void SetCameraFarClip(float far_clip) noexcept;
        void SetCameraProjectionFactor(Angle fov_y, float aspect_ratio, float near_clip = 0.01f, float far_clip = 10000.f) noexcept;
        void SetCameraWorldTransform(Transform to_world) noexcept;
        void CameraRotate(float delta_x, float delta_y) noexcept;
        void CameraMove(Float3 delta) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil