#include "pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"

extern "C" char ddgi_gbuffer_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
double m_time_cnt = 1.f;
}

namespace Pupil::ddgi::gbuffer {
GBufferPass::GBufferPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void GBufferPass::Run() noexcept {
    m_timer.Start();
    {
        if (m_dirty) {
            m_optix_launch_params.camera.SetData(m_optix_scene->camera->GetCudaMemory());
            m_optix_launch_params.random_seed = 0;
            m_dirty = false;
        }

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        ++m_optix_launch_params.random_seed;
    }
    m_timer.Stop();
    m_time_cnt = m_timer.ElapsedMilliseconds();
}

void GBufferPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule(ddgi_gbuffer_pass_embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            // .shadow_miss = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__default" },
            // .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
            // .shadow_grop = { .ch_entry = "__closesthit__shadow",
            //                  .intersect_module = sphere_module }
        };
        pipeline_desc.programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void GBufferPass::SetScene(scene::Scene *scene) noexcept {
    if (m_optix_scene == nullptr)
        m_optix_scene = std::make_unique<optix::Scene>(scene);
    else
        m_optix_scene->ResetScene(scene);

    m_optix_launch_params.config.frame.width = scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = scene->sensor.film.h;

    m_optix_launch_params.random_seed = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc albedo_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_albedo",
        .size = m_output_pixel_num * sizeof(float4)
    };
    m_albedo = buf_mngr->AllocBuffer(albedo_buf_desc);
    m_optix_launch_params.albedo.SetData(m_albedo->cuda_res.ptr, m_output_pixel_num);

    BufferDesc normal_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_normal",
        .size = m_output_pixel_num * sizeof(float4)
    };
    m_normal = buf_mngr->AllocBuffer(normal_buf_desc);
    m_optix_launch_params.normal.SetData(m_normal->cuda_res.ptr, m_output_pixel_num);

    m_optix_launch_params.handle = m_optix_scene->ias_handle;
    m_optix_launch_params.emitters = m_optix_scene->emitters->GetEmitterGroup();

    SetSBT(scene);

    m_dirty = true;
}

void GBufferPass::SetSBT(scene::Scene *scene) noexcept {
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

            // HitGroupDataRecord hit_shadow_data{};
            // hit_shadow_data.program_name = "__closesthit__shadow";
            // hit_shadow_data.data.mat.type = shape.mat.type;
            // desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {
            .program_name = "__miss__default",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_data);
        // decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {
        //     .program_name = "__miss__shadow",
        //     .data = SBTTypes::MissDataType{}
        // };
        // desc.miss_datas.push_back(miss_shadow_data);
    }
    m_optix_pass->InitSBT(desc);
}

void GBufferPass::BindingEventCallback() noexcept {
    EventBinder<ESystemEvent::SceneLoadFinished>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ECanvasEvent::MouseDragging>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;

        m_optix_launch_params.random_seed = 0;

        const struct {
            float x, y;
        } delta = *(decltype(delta) *)p;
        float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
        m_optix_scene->camera->Rotate(delta.x * scale, delta.y * scale);
        m_dirty = true;
    });

    EventBinder<ECanvasEvent::MouseWheel>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;
        m_optix_launch_params.random_seed = 0;

        float delta = *(float *)p;
        m_optix_scene->camera->SetFovDelta(delta);
        m_dirty = true;
    });

    EventBinder<ECanvasEvent::CameraMove>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;
        m_optix_launch_params.random_seed = 0;

        util::Float3 delta = *(util::Float3 *)p;
        m_optix_scene->camera->Move(delta * util::Camera::sensitivity * util::Camera::sensitivity_scale);
        m_dirty = true;
    });
}

void GBufferPass::Inspector() noexcept {
    ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", m_time_cnt, 1000.0f / m_time_cnt);
}
}// namespace Pupil::ddgi::gbuffer