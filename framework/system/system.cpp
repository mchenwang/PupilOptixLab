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

#include <iostream>
#include <format>

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

    m_gui_pass = util::Singleton<GuiPass>::instance();
    m_gui_pass->Init();
    // m_post_pass = util::Singleton<PostProcessPass>::instance();
    // m_post_pass->Init();
    util::Singleton<cuda::Context>::instance()->Init();
    util::Singleton<optix::Context>::instance()->Init();

    EventBinder<SystemEvent::PostProcessFinished>([this]() {
        m_gui_pass->FlipSwapBuffer();
    });
}

void System::Run() noexcept {
    CUDA_SYNC_CHECK();
    for (auto pass : m_pre_passes) pass->BeforeRunning();
    for (auto pass : m_pre_passes) pass->Run();
    for (auto pass : m_pre_passes) pass->AfterRunning();

    // // rendering thread:
    // while (!quit_flag) {
    //     if (render_flag) {
    //         for (auto pass : m_passes) pass->BeforeRunning();
    //         for (auto pass : m_passes) pass->Run();
    //         for (auto pass : m_passes) pass->AfterRunning();
    //     }
    // }
    // // main(application) thread:
    // if (m_gui_pass) {
    //     while (!quit_flag) {
    //         m_gui_pass->Run();
    //     }
    // }

    while (!quit_flag) {
        if (render_flag) {
            for (auto pass : m_passes) pass->BeforeRunning();
            for (auto pass : m_passes) pass->Run();
            for (auto pass : m_passes) pass->AfterRunning();
        }
        if (m_gui_pass) m_gui_pass->Run();
    }
}
void System::Destroy() noexcept {
    // util::Singleton<PostProcessPass>::instance()->Destroy();
    util::Singleton<GuiPass>::instance()->Destroy();
    util::Singleton<cuda::Context>::instance()->Destroy();
    util::Singleton<optix::Context>::instance()->Destroy();
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
        std::cout << std::format("warning: scene file [{}] does not exist.\n", scene_file_path.string());
        return;
    }

    util::Singleton<cuda::CudaTextureManager>::instance()->Clear();

    if (m_scene == nullptr)
        m_scene = std::make_unique<scene::Scene>();
    m_scene->LoadFromXML(scene_file_path);

    for (auto pass : m_pre_passes) pass->SetScene(m_scene.get());
    for (auto pass : m_passes) pass->SetScene(m_scene.get());
    if (m_gui_pass) m_gui_pass->SetScene(m_scene.get());
    // if (m_post_pass) m_post_pass->SetScene(m_scene.get());

    util::Singleton<scene::ShapeDataManager>::instance()->Clear();
    util::Singleton<scene::TextureManager>::instance()->Clear();

    uint64_t size = (static_cast<uint32_t>(m_scene->sensor.film.w)) |
                    (static_cast<uint32_t>(m_scene->sensor.film.h) << 16);
    EventDispatcher<SystemEvent::SceneLoad>(size);
}
}// namespace Pupil