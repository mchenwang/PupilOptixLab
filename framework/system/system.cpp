#include "system.h"

#include "dx12/context.h"
#include "cuda/context.h"
#include "optix/context.h"

#include "cuda/texture.h"
#include "cuda/stream.h"

#include "scene/scene.h"
#include "scene/texture.h"

#include "pass.h"
#include "gui.h"
#include "util/event.h"
#include "util/thread_pool.h"
#include "util/log.h"

#include <iostream>
#include <format>

namespace Pupil {
void System::Init(bool has_window) noexcept {
    util::Singleton<Log>::instance()->Init();
    util::Singleton<util::ThreadPool>::instance()->Init();

    EventBinder<ESystemEvent::Quit>([this](void *) {
        this->quit_flag = true;
    });

    if (!has_window) {
        util::Singleton<cuda::Context>::instance()->Init();
        util::Singleton<optix::Context>::instance()->Init();
        return;
    }

    EventBinder<EWindowEvent::Minimized>([this](void *) {
        this->render_flag = false;
    });
    EventBinder<EWindowEvent::Resize>([this](void *) {
        this->render_flag = true;
    });
    EventBinder<EWindowEvent::Quit>([this](void *) {
        EventDispatcher<ESystemEvent::Quit>();
    });

    m_gui_pass = util::Singleton<GuiPass>::instance();
    m_gui_pass->Init();
    util::Singleton<cuda::Context>::instance()->Init();
    util::Singleton<optix::Context>::instance()->Init();

    EventBinder<ESystemEvent::FrameFinished>([this](void *) {
        m_gui_pass->FlipSwapBuffer();
    });
}

void System::Run() noexcept {
    CUDA_SYNC_CHECK();
    for (auto pass : m_pre_passes) pass->BeforeRunning();
    for (auto pass : m_pre_passes) pass->Run();
    for (auto pass : m_pre_passes) pass->AfterRunning();

    util::Singleton<util::ThreadPool>::instance()->AddTask(
        [&]() {
            while (!quit_flag) {
                if (render_flag) {
                    m_render_timer.Start();
                    for (auto pass : m_passes) pass->BeforeRunning();
                    for (auto pass : m_passes) pass->Run();
                    for (auto pass : m_passes) pass->AfterRunning();
                    m_render_timer.Stop();
                    EventDispatcher<ESystemEvent::FrameFinished>(m_render_timer.ElapsedMilliseconds());
                }
            }
        });

    while (!quit_flag) {
        if (m_gui_pass) m_gui_pass->Run();
    }
}
void System::Destroy() noexcept {
    util::Singleton<util::ThreadPool>::instance()->Destroy();
    util::Singleton<GuiPass>::instance()->Destroy();
    util::Singleton<cuda::Context>::instance()->Destroy();
    util::Singleton<optix::Context>::instance()->Destroy();
    util::Singleton<Log>::instance()->Destroy();
}

void System::AddPass(Pass *pass) noexcept {
    if (pass->tag & EPassTag::Pre)
        m_pre_passes.push_back(pass);
    else
        m_passes.push_back(pass);
    if (m_gui_pass) {
        m_gui_pass->RegisterInspector(
            pass->name,
            [pass]() {
                pass->Inspector();
            });
    }
}

void System::SetScene(std::filesystem::path scene_file_path) noexcept {
    if (!std::filesystem::exists(scene_file_path)) {
        Pupil::Log::Warn("scene file [{}] does not exist.", scene_file_path.string());
        return;
    }

    Pupil::Log::Info("start loading scene [{}].", scene_file_path.string());
    util::Singleton<cuda::CudaTextureManager>::instance()->Clear();

    if (m_scene == nullptr)
        m_scene = std::make_unique<scene::Scene>();
    m_scene->LoadFromXML(scene_file_path);

    for (auto pass : m_pre_passes) pass->SetScene(m_scene.get());
    for (auto pass : m_passes) pass->SetScene(m_scene.get());
    if (m_gui_pass) m_gui_pass->SetScene(m_scene.get());

    util::Singleton<scene::ShapeDataManager>::instance()->Clear();
    util::Singleton<scene::TextureManager>::instance()->Clear();

    this->render_flag = true;
    struct {
        uint32_t w, h;
    } size{ static_cast<uint32_t>(m_scene->sensor.film.w),
            static_cast<uint32_t>(m_scene->sensor.film.h) };
    EventDispatcher<ESystemEvent::SceneLoadFinished>(size);
}

void System::StopRendering() noexcept {
    render_flag = false;
}
void System::RestartRendering() noexcept {
    render_flag = true;
}

}// namespace Pupil