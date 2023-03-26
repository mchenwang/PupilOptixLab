#pragma once

#include "util/util.h"

namespace Pupil {
class Pass;
class GuiPass;

enum class SystemEvent {
    Quit
};

class System : public util::Singleton<System> {
public:
    bool render_flag = true;
    bool quit_flag = false;

    void Init(bool has_window = true) noexcept;
    void Run() noexcept;
    void Destroy() noexcept;

    void AddPass(Pass *) noexcept;

private:
    std::vector<Pass *> m_passes;
    GuiPass *m_gui_pass = nullptr;
};
}// namespace Pupil