#pragma once

#include "util/util.h"
#include "util/timer.h"
#include "scene/scene.h"

#include <filesystem>
#include <memory>

namespace Pupil {
class Pass;
class GuiPass;

enum class ESystemEvent {
    Quit,
    SceneLoadFinished,
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

    void StopRendering() noexcept;
    void RestartRendering() noexcept;

private:
    std::vector<Pass *> m_passes;
    std::vector<Pass *> m_pre_passes;
    GuiPass *m_gui_pass = nullptr;
    std::unique_ptr<scene::Scene> m_scene;
    Timer m_render_timer;
};
}// namespace Pupil