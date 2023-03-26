#include "system.h"

#include "dx12/context.h"
#include "cuda/context.h"
#include "optix/context.h"

#include "gui.h"
#include "event.h"

namespace Pupil {
void System::Init(bool has_window) noexcept {

    EventBinder<SystemEvent::Quit>([this]() {
        this->quit_flag = true;
    });

    if (!has_window) {
        util::Singleton<cuda::Context>::instance()->Init();
        util::Singleton<optix::Context>::instance()->Init();
        return;
    }

    EventBinder<WindowEvent::Minimized>([this]() {
        this->render_flag = false;
    });
    EventBinder<WindowEvent::Resize>([this]() {
        this->render_flag = true;
    });
    EventBinder<WindowEvent::Quit>([this]() {
        EventDispatcher<SystemEvent::Quit>();
    });

    util::Singleton<GuiPass>::instance()->Init();
    util::Singleton<cuda::Context>::instance()->Init();
    util::Singleton<optix::Context>::instance()->Init();

    m_gui_pass = util::Singleton<GuiPass>::instance();
}

void System::Run() noexcept {
    while (!quit_flag) {
        if (render_flag) {
            for (auto pass : m_passes) {
                pass->Run();
            }
        }
        if (m_gui_pass) m_gui_pass->Run();
    }
}
void System::Destroy() noexcept {
    util::Singleton<GuiPass>::instance()->Destroy();
    util::Singleton<cuda::Context>::instance()->Destroy();
    util::Singleton<optix::Context>::instance()->Destroy();
}

void System::AddPass(Pass *pass) noexcept {
    m_passes.push_back(pass);
}
}// namespace Pupil