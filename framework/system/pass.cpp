#include "pass.h"

#include "imgui.h"

namespace Pupil {
    void Pass::Run() noexcept {
        if (!m_enable) return;

        OnRun();
    }

    void Pass::Console() noexcept {
        ImGui::Checkbox("enalbe", &m_enable);
    }
}// namespace Pupil