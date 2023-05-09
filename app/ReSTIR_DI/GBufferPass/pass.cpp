#include "pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "scene/scene.h"
#include "optix/scene/scene.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"
#include "system/type.h"

using namespace Pupil;

extern "C" char g_restir_di_gbuffer_ptx[];

namespace {
double m_time_cost = 0.f;
}

mat4x4 prev_camera_proj_view;

GBufferPass::GBufferPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<GBufferPassSBTType, GBufferPassLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void GBufferPass::Run() noexcept {
    m_timer.Start();
    {
        prev_camera_proj_view = m_params.camera.proj_view;
        if (m_dirty) {
            auto proj = m_world_camera->GetProjectionMatrix();
            auto view = m_world_camera->GetViewMatrix();
            m_params.camera.proj_view = Pupil::ToCudaType(proj * view);
            m_params.camera.view = m_world_camera->GetViewCudaMatrix();
            m_params.camera.camera_to_world = m_world_camera->GetToWorldCudaMatrix();
            m_params.camera.sample_to_camera = m_world_camera->GetSampleToCameraCudaMatrix();

            m_params.random_seed = 0;
            m_dirty = false;
        }
        m_optix_pass->Run(m_params, m_params.frame.width, m_params.frame.height);
        m_optix_pass->Synchronize();

        m_params.random_seed += 3;
    }
    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void GBufferPass::SetScene(Pupil::World *world) noexcept {
    m_world_camera = world->camera.get();

    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;

    m_params.random_seed = 0;
    m_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    m_params.handle = world->optix_scene->ias_handle;

    m_output_pixel_num = m_params.frame.width * m_params.frame.height;
    {
        auto buf_mngr = util::Singleton<BufferManager>::instance();
        BufferDesc desc{
            .type = EBufferType::Cuda,
            .name = std::string{ POS },
            .size = m_output_pixel_num * sizeof(float4)
        };
        m_pos_buf = buf_mngr->AllocBuffer(desc);
        desc.name = std::string{ NORMAL };
        m_nor_buf = buf_mngr->AllocBuffer(desc);
        desc.name = std::string{ ALBEDO };
        m_alb_buf = buf_mngr->AllocBuffer(desc);

        m_params.position.SetData(m_pos_buf->cuda_res.ptr, m_output_pixel_num);
        m_params.normal.SetData(m_nor_buf->cuda_res.ptr, m_output_pixel_num);
        m_params.albedo.SetData(m_alb_buf->cuda_res.ptr, m_output_pixel_num);

        BufferDesc reservoir_buf_desc{
            .type = EBufferType::Cuda,
            .name = "screen reservoir",
            .size = m_output_pixel_num * sizeof(Reservoir)
        };
        auto reservoir_buf = buf_mngr->AllocBuffer(reservoir_buf_desc);
        m_params.reservoirs.SetData(reservoir_buf->cuda_res.ptr, m_output_pixel_num);
    }

    {
        optix::SBTDesc<GBufferPassSBTType> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main",
        };
        {
            int emitter_index_offset = 0;
            using HitGroupDataRecord = optix::ProgDataDescPair<GBufferPassSBTType::HitGroupDataType>;
            for (auto &&shape : world->scene->shapes) {
                HitGroupDataRecord hit_default_data{};
                hit_default_data.program = "__closesthit__default";
                hit_default_data.data.mat.LoadMaterial(shape.mat);
                hit_default_data.data.geo.LoadGeometry(shape);
                if (shape.is_emitter) {
                    hit_default_data.data.emitter_index_offset = emitter_index_offset;
                    emitter_index_offset += shape.sub_emitters_num;
                }

                desc.hit_datas.push_back(hit_default_data);

                HitGroupDataRecord hit_shadow_data{};
                hit_shadow_data.program = "__closesthit__default";
                desc.hit_datas.push_back(hit_shadow_data);
            }
        }
        {
            optix::ProgDataDescPair<GBufferPassSBTType::MissDataType> miss_data = {
                .program = "__miss__default"
            };
            desc.miss_datas.push_back(miss_data);
            optix::ProgDataDescPair<GBufferPassSBTType::MissDataType> miss_shadow_data = {
                .program = "__miss__default"
            };
            desc.miss_datas.push_back(miss_shadow_data);
        }
        {
            auto mat_programs = Pupil::material::GetMaterialProgramDesc();
            for (auto &mat_prog : mat_programs) {
                if (mat_prog.cc_entry) {
                    optix::ProgDataDescPair<GBufferPassSBTType::CallablesDataType> cc_data = {
                        .program = mat_prog.cc_entry
                    };
                    desc.callables_datas.push_back(cc_data);
                }
                if (mat_prog.dc_entry) {
                    optix::ProgDataDescPair<GBufferPassSBTType::CallablesDataType> dc_data = {
                        .program = mat_prog.dc_entry
                    };
                    desc.callables_datas.push_back(dc_data);
                }
            }
        }
        m_optix_pass->InitSBT(desc);
    }

    m_dirty = true;
}

void GBufferPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto rt_module = module_mngr->GetModule(g_restir_di_gbuffer_ptx);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc desc{
            .module_ptr = rt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" }
        };
        pipeline_desc.ray_trace_programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::RayTraceProgramDesc desc{
            .module_ptr = rt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module }
        };
        pipeline_desc.ray_trace_programs.push_back(desc);
    }
    {
        auto mat_programs = Pupil::material::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void GBufferPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void GBufferPass::Inspector() noexcept {
    ImGui::Text("cost: %d ms", (int)m_time_cost);
}
