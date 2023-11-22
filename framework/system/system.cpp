#include "system.h"
#include "world.h"
#include "event.h"
#include "pass.h"
#include "buffer.h"

#include "gui/pass.h"

#include "dx12/context.h"
#include "cuda/context.h"
#include "optix/context.h"

#include "cuda/stream.h"

#include "util/log.h"
#include "util/thread_pool.h"

#include <mutex>
#include <condition_variable>
#include <Windows.h>

namespace Pupil {
    namespace Event {
        using RenderPauseHandler    = Handler0A;
        using RenderContinueHandler = Handler0A;
        using RenderRestartHandler  = Handler0A;
    }// namespace Event

    struct System::Impl {
        std::unique_ptr<std::jthread> render_thread;
        std::unique_ptr<std::jthread> main_thread;

        Gui::Pass*                         gui_pass = nullptr;
        std::vector<std::unique_ptr<Pass>> passes;
        std::vector<std::unique_ptr<Pass>> pre_passes;

        bool         limit_render_frame_rate = false;
        unsigned int max_render_frame_rate   = 60;

        bool system_quit_flag = false;

        bool render_flag         = false;
        bool render_restart_flag = true;

        void BindEvents(System* system, bool has_wnidow) noexcept;
    };

    System::System() noexcept {
        m_impl = new Impl();
    }

    System::~System() noexcept {
        delete m_impl;
    }

    void System::Impl::BindEvents(System* system, bool has_wnidow) noexcept {
        auto event_center = util::Singleton<Event::Center>::instance();

        event_center->BindEvent(
            Event::DispatcherMain, Event::RequestQuit,
            new Event::Handler0A([system, event_center]() {
                event_center->Send(Event::RenderPause);

                system->m_impl->render_thread->request_stop();
                system->m_impl->system_quit_flag = true;
            }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::RenderContinue,
            new Event::RenderContinueHandler([system]() {
                system->m_impl->render_flag = true;
            }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::RenderPause,
            new Event::RenderPauseHandler([system]() {
                system->m_impl->render_flag = false;
            }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::RenderRestart,
            new Event::RenderRestartHandler([system]() {
                system->m_impl->render_restart_flag = true;
                system->m_impl->render_flag         = true;
            }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::RequestSceneLoad,
            new Event::Handler1A<std::string>([system, event_center](const std::string path) {
                event_center->DispatchImmediately(Event::RenderPause);
                event_center->DispatchImmediately(Event::SceneLoading);

                auto world = util::Singleton<World>::instance();
                if (!world->LoadScene(path)) {
                    event_center->Send(Event::RenderContinue);
                    return;
                }

                auto buf_mngr = util::Singleton<BufferManager>::instance();

                BufferDesc default_frame_buffer_desc{
                    .name   = buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME.data(),
                    .flag   = (system->m_impl->gui_pass ? EBufferFlag::AllowDisplay :
                                                          EBufferFlag::None),
                    .width  = static_cast<uint32_t>(world->GetScene()->film_w),
                    .height = static_cast<uint32_t>(world->GetScene()->film_h),

                    .stride_in_byte = sizeof(float) * 4};
                buf_mngr->AllocBuffer(default_frame_buffer_desc);

                util::Singleton<cuda::StreamManager>::instance()->Synchronize(
                    cuda::EStreamTaskType::ShapeUploading |
                    cuda::EStreamTaskType::TextureUploading |
                    cuda::EStreamTaskType::MaterialUploading |
                    cuda::EStreamTaskType::EmitterUploading);

                event_center->Send(Event::SceneReset);
                event_center->Send(Event::RenderRestart);
            }));

        if (!has_wnidow) return;

        event_center->BindEvent(
            Event::DispatcherRender, Gui::Event::WindowMinimized,
            new Event::Handler0A([system, event_center]() {
                event_center->Send(Event::RenderPause);
            }));
    }

    void System::Init(bool has_window) noexcept {
        util::Singleton<Log>::instance()->Init();

        auto event_center = util::Singleton<Event::Center>::instance();
        event_center->RegisterDispatcher(Event::DispatcherMain);
        event_center->RegisterDispatcher(Event::DispatcherRender);

        util::Singleton<util::ThreadPool>::instance()->Init();
        util::Singleton<World>::instance()->Init();

        m_impl->BindEvents(this, has_window);

        if (has_window) {
            m_impl->gui_pass = util::Singleton<Gui::Pass>::instance();
            m_impl->gui_pass->Init();
        }

        util::Singleton<cuda::Context>::instance()->Init();
        util::Singleton<optix::Context>::instance()->Init();
    }

    void System::Run() noexcept {
        auto event_center = util::Singleton<Event::Center>::instance();

        m_impl->render_thread = std::make_unique<std::jthread>([this, event_center](std::stop_token st) {
            auto time_point1 = std::chrono::system_clock::now();
            auto time_point2 = std::chrono::system_clock::now();

            size_t frame_cnt = 0;

            while (!st.stop_requested()) {
                // {
                //     std::unique_lock lock(m_impl->render_mutex);
                //     m_impl->render_cv.wait(lock, [&]() { return m_impl->render_flag; });
                // }

                // if (m_impl->limit_render_frame_rate) {
                //     timeBeginPeriod(1);
                //     time_point1    = std::chrono::system_clock::now();
                //     auto work_time = std::chrono::duration<double, std::milli>(time_point1 - time_point2);

                //     const auto limit = 1000.0 / m_impl->max_render_frame_rate;
                //     if (work_time.count() < limit) {
                //         std::chrono::duration<double, std::milli> delta_ms(limit - work_time.count());

                //         auto delta_ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(delta_ms);
                //         std::this_thread::sleep_for(std::chrono::milliseconds(delta_ms_duration.count()));
                //     }

                //     time_point2 = std::chrono::system_clock::now();
                //     // auto sleep_time = std::chrono::duration<double, std::milli>(time_point2 - time_point1);
                //     // auto time_cost = (work_time + sleep_time).count();
                //     // printf("Sleep Time: %.3lf; Work Time: %.3lf; FPS: %.1f\n", sleep_time.count(), work_time.count(), (float)(1000.f / time_cost));

                //     timeEndPeriod(1);
                // }

                event_center->Dispatch(Event::DispatcherRender);

                if (!m_impl->render_flag) {
                    std::this_thread::yield();
                    continue;
                }

                util::Singleton<World>::instance()->Upload();

                if (m_impl->render_restart_flag) {
                    frame_cnt = 0;
                    for (auto& pass : m_impl->pre_passes)
                        if (pass->IsEnabled())
                            pass->Run();
                }

                for (auto& pass : m_impl->passes)
                    if (pass->IsEnabled())
                        pass->Run();

                util::Singleton<Event::Center>::instance()->Send(Event::FrameDone, {++frame_cnt});
            }

            event_center->Dispatch(Event::DispatcherRender);
        });

        while (!m_impl->system_quit_flag) {
            event_center->Dispatch(Event::DispatcherMain);

            if (m_impl->gui_pass) {

                m_impl->gui_pass->Run();
                // Log::Info("{}", i++);
            }
        }
    }

    void System::SetFrameRateLimit(int limit) noexcept {
        if (limit < 1) {
            m_impl->limit_render_frame_rate = false;
        } else {
            m_impl->limit_render_frame_rate = true;
            m_impl->max_render_frame_rate   = limit;
        }
    }

    void System::AddPass(Pass* pass) noexcept {
        if (!(pass->tag & EPassTag::DisableCustomConsole) && m_impl->gui_pass) {
            m_impl->gui_pass->RegisterConsole(pass->name, [pass]() { pass->Console(); });
        }

        if (pass->tag & EPassTag::Pre) {
            m_impl->pre_passes.emplace_back(pass);
        } else {
            m_impl->passes.emplace_back(pass);
        }
    }

    void System::Destroy() noexcept {
        // m_impl->main_thread->join();

        m_impl->render_thread.reset();
        // m_impl->main_thread.reset();

        m_impl->passes.clear();
        m_impl->pre_passes.clear();

        util::Singleton<util::ThreadPool>::instance()->Destroy();
        util::Singleton<World>::instance()->Destroy();
        util::Singleton<BufferManager>::instance()->Destroy();
        util::Singleton<cuda::Context>::instance()->Destroy();
        util::Singleton<optix::Context>::instance()->Destroy();
        util::Singleton<Gui::Pass>::instance()->Destroy();
        util::Singleton<DirectX::Context>::instance()->Destroy();
        util::Singleton<Event::Center>::instance()->Destroy();
        util::Singleton<Log>::instance()->Destroy();
    }

}// namespace Pupil