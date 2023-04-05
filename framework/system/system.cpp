#include "system.h"
#include "world.h"

#include "dx12/context.h"
#include "cuda/context.h"
#include "optix/context.h"
#include "optix/scene/scene.h"

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
    util::Singleton<World>::instance()->Init();

    EventBinder<ESystemEvent::Quit>([this](void *) {
        this->quit_flag = true;
    });

    EventBinder<ESystemEvent::StartRendering>([this](void *) {
        this->render_flag = true;
    });
    EventBinder<ESystemEvent::StopRendering>([this](void *) {
        this->render_flag = false;
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
    util::Singleton<World>::instance()->Destroy();
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

    auto world = util::Singleton<World>::instance();
    if (!world->LoadScene(scene_file_path)) {
        Pupil::Log::Warn("scene load failed.");
        return;
    }
    auto *p = &world;

    EventDispatcher<ESystemEvent::SceneLoad>(world);

    util::Singleton<scene::ShapeDataManager>::instance()->Clear();
    util::Singleton<scene::TextureManager>::instance()->Clear();

    struct {
        uint32_t w, h;
    } size{ static_cast<uint32_t>(world->scene->sensor.film.w),
            static_cast<uint32_t>(world->scene->sensor.film.h) };
    EventDispatcher<ECanvasEvent::Resize>(size);
    EventDispatcher<ESystemEvent::StartRendering>();
}
}// namespace Pupil