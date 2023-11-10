#pragma once
#include "scene/scene.h"

#include "util/util.h"

#include <filesystem>
#include <memory>

namespace Pupil {
    enum class EWorldEvent {
        CameraChange,
        CameraMove,
        CameraFovChange,
        CameraViewChange,
        RenderInstanceTransform,
        RenderInstanceUpdate,
        RenderInstanceRemove
    };

    class World : public util::Singleton<World> {
    public:
        void Init() noexcept;
        void Destroy() noexcept;

        Scene* GetScene() noexcept;

        bool LoadScene(std::filesystem::path) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil