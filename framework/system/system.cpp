#include "system.h"
#include "world.h"

#include "dx12/context.h"
#include "cuda/context.h"
#include "optix/context.h"

#include "cuda/stream.h"

#include "resource/texture.h"

#include "pass.h"
#include "gui/gui.h"
#include "util/event.h"
#include "util/thread_pool.h"
#include "util/log.h"

#include <iostream>
#include <format>
#include <condition_variable>

namespace {
    bool m_system_run_flag = false;
    bool m_scene_load_flag = false;

    std::mutex m_render_system_mutex;
}// namespace

namespace Pupil {
    void System::Init(bool has_window) noexcept {
        util::Singleton<Log>::instance()->Init();
        util::Singleton<util::ThreadPool>::instance()->Init();
        util::Singleton<World>::instance()->Init();

        EventBinder<ESystemEvent::Quit>([this](void*) {
            this->quit_flag = true;
        });

        EventBinder<ESystemEvent::StartRendering>([this](void*) {
            this->render_flag = true;
            m_scene_load_flag = true;
        });
        EventBinder<ESystemEvent::StopRendering>([this](void*) {
            this->render_flag = false;
        });

        EventBinder<ESystemEvent::Precompute>([this](void*) {
            CUDA_SYNC_CHECK();
            for (auto pass : m_pre_passes) pass->Run();
        });

        if (!has_window) {
            util::Singleton<cuda::Context>::instance()->Init();
            util::Singleton<optix::Context>::instance()->Init();
            return;
        }

        EventBinder<EWindowEvent::Minimized>([this](void*) {
            this->render_flag = false;
        });
        EventBinder<EWindowEvent::Resize>([this](void*) {
            this->render_flag = true;
        });
        EventBinder<EWindowEvent::Quit>([this](void*) {
            EventDispatcher<ESystemEvent::Quit>();
        });

        m_gui_pass = util::Singleton<GuiPass>::instance();
        m_gui_pass->Init();
        util::Singleton<cuda::Context>::instance()->Init();
        util::Singleton<optix::Context>::instance()->Init();

        EventBinder<ESystemEvent::FrameFinished>([this](void*) {
            m_gui_pass->FlipSwapBuffer();
        });
    }

    void System::Run() noexcept {
        m_system_run_flag = true;
        if (m_scene_load_flag) {
            EventDispatcher<ESystemEvent::Precompute>();
            EventDispatcher<ESystemEvent::StartRendering>();
        } else {
            EventDispatcher<ESystemEvent::StopRendering>();
        }

        util::Singleton<util::ThreadPool>::instance()->AddTask(
            [&]() {
                util::Singleton<cuda::StreamManager>::instance()->Synchronize(
                    cuda::EStreamTaskType::BufferCreation |
                    cuda::EStreamTaskType::SBTUploading |
                    cuda::EStreamTaskType::GlobalUploading);

                while (!quit_flag) {
                    if (render_flag) {
                        std::unique_lock render_lock(m_render_system_mutex);
                        m_render_timer.Start();
                        for (auto pass : m_passes) pass->Run();
                        m_render_timer.Stop();
                        EventDispatcher<ESystemEvent::FrameFinished>(m_render_timer.ElapsedMilliseconds());
                    }
                }
            });

        while (!quit_flag) {
            if (m_gui_pass) m_gui_pass->Run();
        }

        util::Singleton<util::ThreadPool>::instance()->JoinAll();
    }

    void System::Destroy() noexcept {
        util::Singleton<util::ThreadPool>::instance()->Destroy();
        util::Singleton<World>::instance()->Destroy();
        util::Singleton<GuiPass>::instance()->Destroy();
        util::Singleton<BufferManager>::instance()->Destroy();
        // util::Singleton<cuda::CudaTextureManager>::instance()->Clear();
        // util::Singleton<cuda::CudaShapeDataManager>::instance()->Clear();
        util::Singleton<cuda::Context>::instance()->Destroy();
        util::Singleton<optix::Context>::instance()->Destroy();
        util::Singleton<DirectX::Context>::instance()->Destroy();
        util::Singleton<Log>::instance()->Destroy();
    }

    void System::AddPass(Pass* pass) noexcept {
        if (pass->tag & EPassTag::Pre)
            m_pre_passes.push_back(pass);
        else
            m_passes.push_back(pass);
    }

    void System::SetScene(std::filesystem::path scene_file_path) noexcept {
        if (!std::filesystem::exists(scene_file_path)) {
            Pupil::Log::Warn("scene file [{}] does not exist.", scene_file_path.string());
            return;
        }

        {
            std::unique_lock render_lock(m_render_system_mutex);

            auto world = util::Singleton<World>::instance();
            if (!world->LoadScene(scene_file_path))
                return;

            auto buf_mngr = util::Singleton<BufferManager>::instance();

            BufferDesc default_frame_buffer_desc{
                .name   = buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME.data(),
                .flag   = (util::Singleton<GuiPass>::instance()->IsInitialized() ?
                               EBufferFlag::AllowDisplay :
                               EBufferFlag::None),
                .width  = static_cast<uint32_t>(world->GetScene()->film_w),
                .height = static_cast<uint32_t>(world->GetScene()->film_h),

                .stride_in_byte = sizeof(float) * 4};
            buf_mngr->AllocBuffer(default_frame_buffer_desc);

            m_scene_load_flag = true;

            util::Singleton<cuda::StreamManager>::instance()->Synchronize(
                cuda::EStreamTaskType::ShapeUploading |
                cuda::EStreamTaskType::TextureUploading |
                cuda::EStreamTaskType::MaterialUploading |
                cuda::EStreamTaskType::EmitterUploading);

            EventDispatcher<ESystemEvent::SceneLoad>(world);
            EventDispatcher<EWorldEvent::CameraChange>();
        }

        this->render_flag = true;

        if (m_system_run_flag) {
            EventDispatcher<ESystemEvent::Precompute>();
            EventDispatcher<ESystemEvent::StartRendering>();
        }
    }
}// namespace Pupil