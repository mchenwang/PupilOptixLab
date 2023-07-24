#pragma once

#include "util/util.h"
#include "camera.h"

#include "resource/scene.h"
#include "optix/scene/scene.h"

#include <filesystem>
#include <memory>

namespace Pupil {
enum class EWorldEvent {
    CameraChange,
    CameraMove,
    CameraFovChange,
    CameraViewChange,
    RenderInstanceUpdate
};

class World : public util::Singleton<World> {
public:
    std::unique_ptr<resource::Scene> scene = nullptr;
    std::unique_ptr<optix::Scene> optix_scene = nullptr;
    std::unique_ptr<CameraHelper> camera = nullptr;

    void Init() noexcept;
    void Destroy() noexcept;

    bool LoadScene(std::filesystem::path) noexcept;

    Pupil::util::Camera &GetUtilCamera() noexcept { return camera->GetUtilCamera(); }
    Pupil::optix::Camera &GetOptixCamera() noexcept { return camera->GetOptixCamera(); }
};
}// namespace Pupil