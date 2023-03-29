#include "pt_pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "scene/scene.h"

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace Pupil::pt {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    auto cuda_stream = cuda_ctx->GetStream(cuda::Context::DEFAULT_STREAM);
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, cuda_stream);
    InitOptixPipeline();
    BindingEventCallback();
}
void PTPass::Run() noexcept {
    m_optix_launch_params.camera.SetData(m_optix_scene->camera->GetCudaMemory());
    // auto backend = util::Singleton<gui::Window>::instance()->GetBackend();
    // params.frame_buffer = reinterpret_cast<float4 *>(backend->GetCurrentFrameResource().src->cuda_buffer_ptr);
    // pass->Run(params, params.config.frame.width, params.config.frame.height);

    m_optix_launch_params.sample_cnt += m_optix_launch_params.config.accumulated_flag;
    ++m_optix_launch_params.random_seed;
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule("path_tracer/pt_main.ptx");

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

void PTPass::SetScene(scene::Scene *scene) noexcept {
    if (m_optix_scene == nullptr)
        m_optix_scene = std::make_unique<optix::Scene>(scene);
    else
        m_optix_scene->ResetScene(scene);

    m_optix_launch_params.config.frame.width = scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = scene->sensor.film.h;
    m_optix_launch_params.config.max_depth = scene->integrator.max_depth;
    m_optix_launch_params.config.accumulated_flag = true;
    m_optix_launch_params.config.use_tone_mapping = false;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.sample_cnt = 0;

    CUDA_FREE(m_accum_buffer);
    size_t pixel_num = m_optix_launch_params.config.frame.width *
                       m_optix_launch_params.config.frame.height;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&m_accum_buffer),
        pixel_num * sizeof(float4)));

    m_optix_launch_params.accum_buffer.SetData(m_accum_buffer, pixel_num);

    m_optix_launch_params.frame_buffer.SetData(0, 0);
    m_optix_launch_params.handle = m_optix_scene->ias_handle;
    m_optix_launch_params.emitters = m_optix_scene->emitters->GetEmitterGroup();
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
}

void PTPass::Inspector() noexcept {
    ImGui::SeparatorText("info");
    {
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("Window Size(width x height): %d x %d", g_window_w, g_window_h);
    }
}
}// namespace Pupil::pt