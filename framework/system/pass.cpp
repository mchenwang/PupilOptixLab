#include "pass.h"

#include "imgui.h"

namespace Pupil {
void Pass::Run() noexcept {
    if (!m_enable) return;

    m_timer.Start();
    OnRun();
    m_timer.Stop();
    m_last_exec_time = m_timer.ElapsedMilliseconds();
}

void Pass::Inspector() noexcept {
    ImGui::Checkbox("enalbe", &m_enable);
    ImGui::Text("time cost: %.3lf ms", m_last_exec_time);
}
}// namespace Pupil