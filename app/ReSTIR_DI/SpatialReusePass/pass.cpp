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

extern "C" char g_restir_di_spat_reuse_ptx[];

namespace {
double m_time_cost = 0.f;
bool m_flag = false;
int m_spatial_radius = 30;
}// namespace

SpatialReusePass::SpatialReusePass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SpatialReusePassSBTType, SpatialReusePassLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void SpatialReusePass::Run() noexcept {
    m_timer.Start();
    if (m_flag) {
        if (m_dirty) {
            auto cam_pos = m_camera->GetPosition();
            m_params.camera.pos = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);

            m_params.spatial_radius = m_spatial_radius;
        }
        m_optix_pass->Run(m_params, m_params.frame.width, m_params.frame.height);
        m_optix_pass->Synchronize();

        m_params.random_seed += 3;
    } else {
        auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
        auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
        auto final_reservoir_buf = buf_mngr->GetBuffer("final screen reservoir");

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(final_reservoir_buf->cuda_res.ptr),
            reinterpret_cast<void *>(reservoir_buf->cuda_res.ptr),
            m_output_pixel_num * sizeof(Reservoir), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));
        CUDA_CHECK(cudaStreamSynchronize(m_stream->GetStream()));
    }
    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void SpatialReusePass::SetScene(Pupil::World *world) noexcept {
    m_camera = &world->GetUtilCamera();
    auto cam_pos = m_camera->GetPosition();
    m_params.camera.pos = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);

    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;

    m_output_pixel_num = m_params.frame.width * m_params.frame.height;
    auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
    auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
    m_params.reservoirs.SetData(reservoir_buf->cuda_res.ptr, m_output_pixel_num);

    BufferDesc final_reservoir_buf_desc{
        .type = EBufferType::Cuda,
        .name = "final screen reservoir",
        .size = m_output_pixel_num * sizeof(Reservoir)
    };
    auto final_reservoir_buf = buf_mngr->AllocBuffer(final_reservoir_buf_desc);
    m_params.final_reservoirs.SetData(final_reservoir_buf->cuda_res.ptr, m_output_pixel_num);

    auto pos_buf = buf_mngr->GetBuffer(GBufferPass::POS);
    m_params.position.SetData(pos_buf->cuda_res.ptr, m_output_pixel_num);
    auto alb_buf = buf_mngr->GetBuffer(GBufferPass::ALBEDO);
    m_params.albedo.SetData(alb_buf->cuda_res.ptr, m_output_pixel_num);
    auto nor_buf = buf_mngr->GetBuffer(GBufferPass::NORMAL);
    m_params.normal.SetData(nor_buf->cuda_res.ptr, m_output_pixel_num);

    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;

    m_params.random_seed = 2;

    {
        optix::SBTDesc<SpatialReusePassSBTType> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main",
        };
        m_optix_pass->InitSBT(desc);
    }

    m_dirty = true;
}

void SpatialReusePass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto rt_module = module_mngr->GetModule(g_restir_di_spat_reuse_ptx);

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

void SpatialReusePass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void SpatialReusePass::Inspector() noexcept {
    ImGui::Text("cost: %d ms", (int)m_time_cost);
    ImGui::Checkbox("use Spatial reuse", &m_flag);
    ImGui::InputInt("spatial radius", &m_spatial_radius, 1, 5);
    m_spatial_radius = clamp(m_spatial_radius, 0, 50);
    if (m_params.spatial_radius != m_spatial_radius) {
        m_dirty = true;
    }
}
