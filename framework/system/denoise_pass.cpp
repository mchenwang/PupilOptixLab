#include "denoise_pass.h"
#include "system/system.h"
#include "util/event.h"

#include "imgui.h"
#include "system/gui/gui.h"

#include <atomic>

namespace {
uint32_t m_denoiser_mode = Pupil::optix::Denoiser::EMode::None;
bool m_mode_dirty = false;

uint32_t m_film_w = 0;
uint32_t m_film_h = 0;
bool m_film_dirty = false;

double m_time_cost = 0.;

uint32_t m_tile_w = 500;
uint32_t m_tile_h = 500;
bool m_tile_dirty = false;

Pupil::optix::Denoiser::ExecutionData m_data;
}// namespace

namespace Pupil {
bool DenoisePass::s_enabled_flag = false;

DenoisePass::DenoisePass(const Config &config, std::string_view name) noexcept
    : Pass(name), m_config(config) {
    m_stream = std::make_unique<cuda::Stream>();

    m_denoiser_mode = (config.use_albedo ? optix::Denoiser::EMode::UseAlbedo : 0) |
                      (config.use_normal ? optix::Denoiser::EMode::UseNormal : 0);

    m_denoiser = std::make_unique<optix::Denoiser>(m_denoiser_mode, m_stream.get());

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });

    s_enabled_flag = config.default_enable;

    if (!s_enabled_flag) {
        EventDispatcher<ECanvasEvent::Display>(std::string_view{ config.noise_name });
    }
}

void DenoisePass::OnRun() noexcept {
    if (!s_enabled_flag) return;

    m_timer.Start();

    if (m_mode_dirty) {
        m_denoiser->SetMode(m_denoiser_mode);
        m_film_dirty = true;
        m_mode_dirty = false;
    }

    if (m_film_dirty) {
        if (m_tile_dirty) {
            m_denoiser->SetTile(m_tile_w, m_tile_h);
            m_tile_dirty = false;
        }
        m_denoiser->Setup(m_film_w, m_film_h);
        m_film_dirty = false;
    }

    m_denoiser->Execute(m_data);
    m_stream->Synchronize();

    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void DenoisePass::SetScene(world::World *world) noexcept {
    if (world->scene->sensor.film.w != m_film_w || world->scene->sensor.film.h != m_film_h) {
        m_film_w = world->scene->sensor.film.w;
        m_film_h = world->scene->sensor.film.h;
        m_film_dirty = true;
    }
    m_denoiser->tile_w = m_tile_w;
    m_denoiser->tile_h = m_tile_h;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    m_data.input = buf_mngr->GetBuffer(m_config.noise_name)->cuda_ptr;
    m_data.albedo = buf_mngr->GetBuffer(m_config.albedo_name)->cuda_ptr;
    m_data.normal = buf_mngr->GetBuffer(m_config.normal_name)->cuda_ptr;
    // m_data.motion_vector = buf_mngr->GetBuffer("motion vector")->cuda_ptr;
    m_data.prev_output = m_data.input;
    m_data.output = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME)->cuda_ptr;
}

void DenoisePass::Inspector() noexcept {
    ImGui::Checkbox("enable", &s_enabled_flag);
    ImGui::Text("cost: %.3lf ms", m_time_cost);
    uint32_t mode = m_denoiser_mode;

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

    if (mode != m_denoiser_mode) {
        m_denoiser_mode = mode;
        m_mode_dirty = true;
    }
}
}// namespace Pupil