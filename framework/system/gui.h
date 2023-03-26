#pragma once

#include "util/util.h"
#include "pass.h"

#include <unordered_map>
#include <functional>

namespace Pupil {
enum class WindowEvent {
    Quit,
    Resize,
    Minimized,
    // MouseLeftButtonMove,
    // MouseRightButtonMove,
    // MouseWheel,
    // KeyboardMove
};

class GuiPass : public Pass, public util::Singleton<GuiPass> {
public:
    GuiPass() noexcept : Pass("GUI") {}

    virtual void Run() noexcept override;
    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    virtual void InitScene(void *scene) noexcept {}

    void Init() noexcept;
    void Destroy() noexcept;
    void Resize(uint32_t, uint32_t) noexcept;
    void AdjustWindowSize() noexcept;

    using CustomGui = std::function<void()>;
    void RegisterGui(std::string_view, CustomGui &&) noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

protected:
    void OnDraw() noexcept;

    std::unordered_map<std::string, CustomGui> m_guis;
    bool m_init_flag = false;
};
}// namespace Pupil