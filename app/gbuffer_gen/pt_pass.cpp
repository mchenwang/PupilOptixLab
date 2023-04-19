#include "pt_pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"

extern "C" char embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
int m_max_depth;
double m_time_cnt = 1.;

int m_show_type = 0;
}// namespace

namespace Pupil::pt {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void PTPass::Run() noexcept {
    m_timer.Start();
    {
        if (m_dirty) {
            m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());
            m_optix_launch_params.config.max_depth = m_max_depth;
            m_optix_launch_params.sample_cnt = 0;
            m_optix_launch_params.random_seed = 0;
            m_dirty = false;
        }

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        ++m_optix_launch_params.sample_cnt;
        ++m_optix_launch_params.random_seed;
    }
    {
        m_timer.Stop();
        m_time_cnt = m_timer.ElapsedMilliseconds();
        Pupil::Log::Info("time cost: {}ms", m_time_cnt);
        Pupil::EventDispatcher<pt::EPTEvent::Finished>();
    }
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule(embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            .shadow_miss = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__default" },
            .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }
    {
        // for sphere geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
            .shadow_grop = { .ch_entry = "__closesthit__shadow",
                             .intersect_module = sphere_module }
        };
        pipeline_desc.programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void PTPass::SetScene(World *world) noexcept {
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = 1920;
    m_optix_launch_params.config.frame.height = 1080;
    m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;

    auto proj = m_world_camera->GetUtilCamera().GetProjectionMatrix();
    auto view = m_world_camera->GetUtilCamera().GetViewMatrix();
    util::Mat4 proj_view = proj * view;
    m_optix_launch_params.camera_proj_view_mat.r0 = make_float4(proj_view.r0.x, proj_view.r0.y, proj_view.r0.z, proj_view.r0.w);
    m_optix_launch_params.camera_proj_view_mat.r1 = make_float4(proj_view.r1.x, proj_view.r1.y, proj_view.r1.z, proj_view.r1.w);
    m_optix_launch_params.camera_proj_view_mat.r2 = make_float4(proj_view.r2.x, proj_view.r2.y, proj_view.r2.z, proj_view.r2.w);
    m_optix_launch_params.camera_proj_view_mat.r3 = make_float4(proj_view.r3.x, proj_view.r3.y, proj_view.r3.z, proj_view.r3.w);

    m_optix_launch_params.camera_proj_mat.r0 = make_float4(proj.r0.x, proj.r0.y, proj.r0.z, proj.r0.w);
    m_optix_launch_params.camera_proj_mat.r1 = make_float4(proj.r1.x, proj.r1.y, proj.r1.z, proj.r1.w);
    m_optix_launch_params.camera_proj_mat.r2 = make_float4(proj.r2.x, proj.r2.y, proj.r2.z, proj.r2.w);
    m_optix_launch_params.camera_proj_mat.r3 = make_float4(proj.r3.x, proj.r3.y, proj.r3.z, proj.r3.w);

    m_optix_launch_params.camera_view_mat.r0 = make_float4(view.r0.x, view.r0.y, view.r0.z, view.r0.w);
    m_optix_launch_params.camera_view_mat.r1 = make_float4(view.r1.x, view.r1.y, view.r1.z, view.r1.w);
    m_optix_launch_params.camera_view_mat.r2 = make_float4(view.r2.x, view.r2.y, view.r2.z, view.r2.w);
    m_optix_launch_params.camera_view_mat.r3 = make_float4(view.r3.x, view.r3.y, view.r3.z, view.r3.w);

    m_max_depth = m_optix_launch_params.config.max_depth;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.sample_cnt = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc desc{
        .type = EBufferType::Cuda,
        .name = "albedo",
        .size = m_output_pixel_num * sizeof(float4)
    };
    {
        auto buffer = buf_mngr->AllocBuffer(desc);
        m_optix_launch_params.albedo.SetData(buffer->cuda_res.ptr, m_output_pixel_num);
    }
    {
        desc.name = "normal";
        auto buffer = buf_mngr->AllocBuffer(desc);
        m_optix_launch_params.normal.SetData(buffer->cuda_res.ptr, m_output_pixel_num);
    }
    {
        desc.name = "depth";
        auto buffer = buf_mngr->AllocBuffer(desc);
        m_optix_launch_params.depth.SetData(buffer->cuda_res.ptr, m_output_pixel_num);
    }

    m_optix_launch_params.frame_buffer.SetData(0, 0);

    m_optix_launch_params.handle = world->optix_scene->ias_handle;
    m_optix_launch_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    SetSBT(world->scene.get());

    m_dirty = true;
}

void PTPass::SetSBT(scene::Scene *scene) noexcept {
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program_name = "__raygen__main",
        .data = SBTTypes::RayGenDataType{}
    };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = decltype(desc)::Pair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : scene->shapes) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program_name = "__closesthit__default";
            hit_default_data.data.mat.LoadMaterial(shape.mat);
            hit_default_data.data.geo.LoadGeometry(shape);
            if (shape.is_emitter) {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += shape.sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {
            .program_name = "__miss__default",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_data);
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {
            .program_name = "__miss__shadow",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_shadow_data);
    }
    m_optix_pass->InitSBT(desc);
}

void PTPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void PTPass::Inspector() noexcept {
    /*constexpr auto show_type = std::array{ "pt result", "albedo", "normal" };
    ImGui::Combo("result", &m_show_type, show_type.data(), (int)show_type.size());

    ImGui::InputInt("spp", &m_spp);
    if (m_spp < 1) m_spp = 1;
    if (m_optix_launch_params.spp != m_spp) {
        m_dirty = true;
    }
    ImGui::InputInt("max trace depth", &m_max_depth);
    m_max_depth = clamp(m_max_depth, 1, 128);
    if (m_optix_launch_params.config.max_depth != m_max_depth) {
        m_dirty = true;
    }*/
    ImGui::Text("sample count: %d", m_optix_launch_params.sample_cnt);
    ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", m_time_cnt, 1000.0f / m_time_cnt);
}
}// namespace Pupil::pt