#pragma once

#include "util/util.h"

#include <filesystem>
#include <memory>

namespace Pupil {
namespace scene {
class Scene;
}
namespace optix {
class Scene;
class CameraHelper;
}// namespace optix

enum class EWorldEvent {
    CameraChange,
    CameraMove,
    CameraFovChange,
    CameraViewChange,
};

class World : public util::Singleton<World> {
public:
    std::unique_ptr<scene::Scene> scene = nullptr;
    std::unique_ptr<optix::Scene> optix_scene = nullptr;

    void Init() noexcept;
    void Destroy() noexcept;

    bool LoadScene(std::filesystem::path) noexcept;
};
}// namespace Pupil