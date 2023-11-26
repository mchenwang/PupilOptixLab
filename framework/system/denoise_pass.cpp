#include "denoise_pass.h"
#include "world.h"
#include "system.h"
#include "event.h"
#include "buffer.h"
#include "scene/scene.h"
#include "util/timer.h"

#include "optix/denoiser.h"

#include "cuda/stream.h"

#include "imgui.h"
#include "system/profiler.h"
#include "system/gui/pass.h"

#include <atomic>
#include <memory>
#include <mutex>

namespace Pupil {
    bool DenoisePass::s_enabled_flag = false;

    struct DenoisePass::Impl {
        util::CountableRef<cuda::Stream> stream;
        std::unique_ptr<optix::Denoiser> denoiser;

        Config config;

        Timer* timer;

        uint32_t denoiser_mode = Pupil::optix::Denoiser::EMode::None;
        bool     mode_dirty    = false;

        uint32_t film_w     = 0;
        uint32_t film_h     = 0;
        bool     film_dirty = false;

        uint32_t tile_w     = 500;
        uint32_t tile_h     = 500;
        bool     tile_dirty = false;

        Pupil::optix::Denoiser::ExecutionData data;
    };

    DenoisePass::DenoisePass(const Config& config, std::string_view name) noexcept
        : Pass(name) {
        m_impl         = new Impl();
        m_impl->config = config;
        m_impl->stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::Render);

        m_impl->timer = util::Singleton<Profiler>::instance()
                            ->AllocTimer(name, m_impl->stream, 60);

        m_impl->denoiser_mode = (config.use_albedo ? optix::Denoiser::EMode::UseAlbedo : 0) |
                                (config.use_normal ? optix::Denoiser::EMode::UseNormal : 0);

        m_impl->denoiser = std::make_unique<optix::Denoiser>(m_impl->stream, m_impl->denoiser_mode);

        auto event_center = util::Singleton<Pupil::Event::Center>::instance();
        event_center->BindEvent(
            Event::DispatcherRender, Event::SceneReset,
            new Event::Handler0A([this]() {
                auto scene = util::Singleton<World>::instance()->GetScene();
                if (scene->film_w != m_impl->film_w || scene->film_h != m_impl->film_h) {
                    m_impl->film_w     = scene->film_w;
                    m_impl->film_h     = scene->film_h;
                    m_impl->film_dirty = true;
                }
                m_impl->denoiser->SetTile(m_impl->tile_w, m_impl->tile_h);

                auto buf_mngr = util::Singleton<BufferManager>::instance();

                m_impl->data.input  = buf_mngr->GetBuffer(m_impl->config.noise_name)->cuda_ptr;
                m_impl->data.albedo = buf_mngr->GetBuffer(m_impl->config.albedo_name)->cuda_ptr;
                m_impl->data.normal = buf_mngr->GetBuffer(m_impl->config.normal_name)->cuda_ptr;
                // m_impl->data.motion_vector = buf_mngr->GetBuffer("motion vector")->cuda_ptr;
                m_impl->data.prev_output = m_impl->data.input;
                m_impl->data.output      = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME)->cuda_ptr;
            }));

        s_enabled_flag = config.default_enable;

        if (!s_enabled_flag) {
            event_center->Send(Gui::Event::CanvasDisplayTargetChange, {config.noise_name});
        }
    }

    DenoisePass::~DenoisePass() noexcept {
        delete m_impl;
        m_impl = nullptr;
    }

    void DenoisePass::OnRun() noexcept {
        if (!s_enabled_flag) return;

        if (m_impl->mode_dirty) {
            m_impl->denoiser->SetMode(m_impl->denoiser_mode);
            m_impl->film_dirty = true;
            m_impl->mode_dirty = false;
        }

        if (m_impl->film_dirty) {
            if (m_impl->tile_dirty) {
                m_impl->denoiser->SetTile(m_impl->tile_w, m_impl->tile_h);
                m_impl->tile_dirty = false;
            }
            m_impl->denoiser->Setup(m_impl->film_w, m_impl->film_h);
            m_impl->film_dirty = false;
        }
        m_impl->timer->Start();
        m_impl->denoiser->Execute(m_impl->data);
        m_impl->timer->Stop();
    }

    void DenoisePass::Console() noexcept {
        ImGui::Checkbox("enable", &s_enabled_flag);
        util::Singleton<Profiler>::instance()->ShowPlot(name);

        uint32_t mode = m_impl->denoiser_mode;
        if (bool albedo = mode & optix::Denoiser::EMode::UseAlbedo;
            ImGui::Checkbox("use albedo", &albedo)) {
            mode ^= optix::Denoiser::EMode::UseAlbedo;
        }
        if (bool normal = mode & optix::Denoiser::EMode::UseNormal;
            ImGui::Checkbox("use normal", &normal)) {
            mode ^= optix::Denoiser::EMode::UseNormal;
        }
        // if (bool temporal = mode & optix::Denoiser::EMode::UseTemporal;
        //     ImGui::Checkbox("use temporal", &temporal)) {
        //     mode ^= optix::Denoiser::EMode::UseTemporal;
        // }
        // if (bool tiled = mode & optix::Denoiser::EMode::Tiled;
        //     ImGui::Checkbox("use tile", &tiled)) {
        //     mode ^= optix::Denoiser::EMode::Tiled;
        // }

        if (mode != m_impl->denoiser_mode) {
            m_impl->denoiser_mode = mode;
            m_impl->mode_dirty    = true;
        }
    }
}// namespace Pupil