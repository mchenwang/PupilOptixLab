#include "pass.h"
#include "../GBufferPass/pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "scene/scene.h"
#include "optix/scene/scene.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"

using namespace Pupil;

extern "C" char g_restir_di_temp_reuse_ptx[];

namespace {
double m_time_cost = 0.f;
bool m_flag = true;
mat4x4 m_camera_proj_view;
}// namespace

extern mat4x4 prev_camera_proj_view;

TemporalReusePass::TemporalReusePass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<TemporalReusePassSBTType, TemporalReusePassLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void TemporalReusePass::Run() noexcept {
    m_timer.Start();
    if (m_flag) {
        m_params.camera.prev_proj_view = prev_camera_proj_view;
        m_optix_pass->Run(m_params, m_params.frame.width, m_params.frame.height);
        auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
        auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
        auto prev_reservoir_buf = buf_mngr->GetBuffer("prev screen reservoir");

        auto pos_buf = buf_mngr->GetBuffer(GBufferPass::POS);
        auto prev_pos_buf = buf_mngr->GetBuffer("prev gbuffer position");

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(prev_reservoir_buf->cuda_res.ptr),
            reinterpret_cast<void *>(reservoir_buf->cuda_res.ptr),
            m_output_pixel_num * sizeof(Reservoir), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(prev_pos_buf->cuda_res.ptr),
            reinterpret_cast<void *>(pos_buf->cuda_res.ptr),
            m_output_pixel_num * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));

        m_params.prev_frame_reservoirs.SetData(prev_reservoir_buf->cuda_res.ptr, m_output_pixel_num);
        m_params.prev_position.SetData(prev_pos_buf->cuda_res.ptr, m_output_pixel_num);

        // if (m_dirty) {
        //     auto proj = m_camera->GetProjectionMatrix();
        //     auto view = m_camera->GetViewMatrix();
        //     m_camera_proj_view = Pupil::ToCudaType(proj * view);
        // }
        m_optix_pass->Synchronize();

        m_params.random_seed += 3;
    }
    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void TemporalReusePass::SetScene(Pupil::World *world) noexcept {
    m_camera = &world->GetUtilCamera();

    auto proj = m_camera->GetProjectionMatrix();
    auto view = m_camera->GetViewMatrix();
    m_camera_proj_view = Pupil::ToCudaType(proj * view);

    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;

    m_output_pixel_num = m_params.frame.width * m_params.frame.height;
    auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
    auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
    m_params.reservoirs.SetData(reservoir_buf->cuda_res.ptr, m_output_pixel_num);

    BufferDesc prev_reservoir_buf_desc{
        .type = EBufferType::Cuda,
        .name = "prev screen reservoir",
        .size = m_output_pixel_num * sizeof(Reservoir)
    };
    auto prev_reservoir_buf = buf_mngr->AllocBuffer(prev_reservoir_buf_desc);
    m_params.prev_frame_reservoirs.SetData(prev_reservoir_buf->cuda_res.ptr, m_output_pixel_num);

    BufferDesc prev_position_buf_desc{
        .type = EBufferType::Cuda,
        .name = "prev gbuffer position",
        .size = m_output_pixel_num * sizeof(float4)
    };
    auto prev_position_buf = buf_mngr->AllocBuffer(prev_position_buf_desc);
    m_params.prev_position.SetData(prev_position_buf->cuda_res.ptr, m_output_pixel_num);

    auto pos_buf = buf_mngr->GetBuffer(GBufferPass::POS);
    m_params.position.SetData(pos_buf->cuda_res.ptr, m_output_pixel_num);

    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;

    m_params.random_seed = 1;

    {
        optix::SBTDesc<TemporalReusePassSBTType> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main",
        };
        m_optix_pass->InitSBT(desc);
    }

    m_dirty = true;
}

void TemporalReusePass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto rt_module = module_mngr->GetModule(g_restir_di_temp_reuse_ptx);

    optix::PipelineDesc pipeline_desc;
    {
        optix::RayTraceProgramDesc desc{
            .module_ptr = rt_module,
            .ray_gen_entry = "__raygen__main"
        };
        pipeline_desc.ray_trace_programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void TemporalReusePass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        // m_dirty = true;
        if (m_camera) {
            auto proj = m_camera->GetProjectionMatrix();
            auto view = m_camera->GetViewMatrix();
            m_camera_proj_view = Pupil::ToCudaType(proj * view);
        }
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void TemporalReusePass::Inspector() noexcept {
    ImGui::Text("cost: %d ms", (int)m_time_cost);
    ImGui::Checkbox("use temporal reuse", &m_flag);
}
