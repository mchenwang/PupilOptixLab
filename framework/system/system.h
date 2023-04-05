#pragma once

#include "util/util.h"
#include "util/timer.h"

#include <filesystem>
#include <memory>

namespace Pupil {
class Pass;
class GuiPass;

namespace optix {
class Scene;
}

enum class ESystemEvent {
    Quit,
    Precompute,
    StartRendering,
    StopRendering,
    SceneLoad,
    FrameFinished
};

class System : public util::Singleton<System> {
public:
    bool render_flag = true;
    bool quit_flag = false;

    void Init(bool has_window = true) noexcept;
    void Run() noexcept;
    void Destroy() noexcept;

    void AddPass(Pass *) noexcept;
    void SetScene(std::filesystem::path) noexcept;
    [[nodiscard]] optix::Scene *GetOptixScene() noexcept;

private:
    std::vector<Pass *> m_passes;
    std::vector<Pass *> m_pre_passes;
    GuiPass *m_gui_pass = nullptr;
    Timer m_render_timer;
};
}// namespace Pupil