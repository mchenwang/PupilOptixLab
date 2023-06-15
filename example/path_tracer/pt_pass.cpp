#include "pt_pass.h"
#include "imgui.h"

#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"
#include "system/world.h"

extern "C" char embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
int m_max_depth;
bool m_accumulated_flag;
bool m_use_path_guiding;
}// namespace

namespace Pupil::pt {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
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
            m_optix_launch_params.config.accumulated_flag = m_accumulated_flag;
            m_optix_launch_params.sample_cnt = 0;
            m_optix_launch_params.random_seed = 0;
            m_dirty = false;
        }

        auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
        m_optix_launch_params.frame_buffer.SetData(frame_buffer.cuda_ptr, m_output_pixel_num);

        // save the transformation for temporal reuse
        mat4x4 cur_proj_view_mat;
        auto proj = m_world_camera->GetUtilCamera().GetProjectionMatrix();
        auto view = m_world_camera->GetUtilCamera().GetViewMatrix();
        util::Mat4 proj_view = proj * view;
        cur_proj_view_mat.r0 = make_float4(proj_view.r0.x, proj_view.r0.y, proj_view.r0.z, proj_view.r0.w);
        cur_proj_view_mat.r1 = make_float4(proj_view.r1.x, proj_view.r1.y, proj_view.r1.z, proj_view.r1.w);
        cur_proj_view_mat.r2 = make_float4(proj_view.r2.x, proj_view.r2.y, proj_view.r2.z, proj_view.r2.w);
        cur_proj_view_mat.r3 = make_float4(proj_view.r3.x, proj_view.r3.y, proj_view.r3.z, proj_view.r3.w);

        m_optix_launch_params.update_pass = false;
        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        if (m_optix_launch_params.config.use_path_guiding) {
            // update the models
            m_optix_launch_params.update_pass = true;
            m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,m_optix_launch_params.config.frame.height);
            m_optix_pass->Synchronize();

            // swap the model buffer
            std::swap(m_optix_launch_params.pre_model_buffer, m_optix_launch_params.new_model_buffer);

            m_optix_launch_params.pre_proj_view_mat = cur_proj_view_mat;
        }

        m_optix_launch_params.sample_cnt += m_optix_launch_params.config.accumulated_flag;
        ++m_optix_launch_params.random_seed;
    }
    m_timer.Stop();
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .miss_entry = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__shadow" },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {
        // for sphere geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__shadow",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }
    {
        // optix::CallableProgramDesc desc{
        //     .module_ptr = pt_module,
        //     .cc_entry = nullptr,
        //     .dc_entry = "__direct_callable__diffuse_sample",
        // };
        // pipeline_desc.callable_programs.push_back(desc);
        auto mat_programs = Pupil::material::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void PTPass::SetScene(World *world) noexcept {
    m_world_camera = world->camera.get();

    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;
    m_optix_launch_params.config.accumulated_flag = true;
    m_optix_launch_params.config.use_path_guiding = false;

    m_max_depth = m_optix_launch_params.config.max_depth;
    m_accumulated_flag = m_optix_launch_params.config.accumulated_flag;
    m_use_path_guiding = m_optix_launch_params.config.use_path_guiding;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.sample_cnt = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;
    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc desc4{
        .type = EBufferType::Cuda,
        .name = "pt accum buffer",
        .size = m_output_pixel_num * sizeof(float4)
    };
    m_accum_buffer = buf_mngr->AllocBuffer(desc4);
    m_optix_launch_params.accum_buffer.SetData(m_accum_buffer->cuda_res.ptr, m_output_pixel_num);

    BufferDesc desc3{
        .type = EBufferType::Cuda,
        .size = m_output_pixel_num * sizeof(float3)
    };
    desc3.name = "pt normal buffer";
    m_optix_launch_params.normal_buffer.SetData(buf_mngr->AllocBuffer(desc3)->cuda_res.ptr, m_output_pixel_num);
    desc3.name = "pt position buffer";
    m_optix_launch_params.position_buffer.SetData(buf_mngr->AllocBuffer(desc3)->cuda_res.ptr, m_output_pixel_num);
    desc3.name = "pt target buffer";
    m_optix_launch_params.target_buffer.SetData(buf_mngr->AllocBuffer(desc3)->cuda_res.ptr, m_output_pixel_num);

    BufferDesc desc1{
        .type = EBufferType::Cuda,
        .size = m_output_pixel_num * sizeof(float)
    };
    desc1.name = "pt depth buffer";
    m_optix_launch_params.depth_buffer.SetData(buf_mngr->AllocBuffer(desc1)->cuda_res.ptr, m_output_pixel_num);
    desc1.name = "pt pdf buffer";
    m_optix_launch_params.pdf_buffer.SetData(buf_mngr->AllocBuffer(desc1)->cuda_res.ptr, m_output_pixel_num);
    desc1.name = "pt radiance buffer";
    m_optix_launch_params.radiance_buffer.SetData(buf_mngr->AllocBuffer(desc1)->cuda_res.ptr, m_output_pixel_num);

    BufferDesc descM{
        .type = EBufferType::Cuda,
        .size = m_output_pixel_num * sizeof(vMF)
    };
    descM.name = "pt pre model buffer";
    m_optix_launch_params.pre_model_buffer.SetData(buf_mngr->AllocBuffer(descM)->cuda_res.ptr, m_output_pixel_num);
    descM.name = "pt new model buffer";
    m_optix_launch_params.new_model_buffer.SetData(buf_mngr->AllocBuffer(descM)->cuda_res.ptr, m_output_pixel_num);

    m_optix_launch_params.frame_buffer.SetData(0, 0);
    m_optix_launch_params.handle = world->optix_scene->ias_handle;
    m_optix_launch_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    SetSBT(world->scene.get());

    m_dirty = true;
}

void PTPass::SetSBT(scene::Scene *scene) noexcept {
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program = "__raygen__main"
    };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : scene->shapes) {
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
            hit_shadow_data.program = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
            .program = "__miss__default"
        };
        desc.miss_datas.push_back(miss_data);
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_shadow_data = {
            .program = "__miss__shadow"
        };
        desc.miss_datas.push_back(miss_shadow_data);
    }
    {
        auto mat_programs = Pupil::material::GetMaterialProgramDesc();
        for (auto &mat_prog : mat_programs) {
            if (mat_prog.cc_entry) {
                optix::ProgDataDescPair<SBTTypes::CallablesDataType> cc_data = {
                    .program = mat_prog.cc_entry
                };
                desc.callables_datas.push_back(cc_data);
            }
            if (mat_prog.dc_entry) {
                optix::ProgDataDescPair<SBTTypes::CallablesDataType> dc_data = {
                    .program = mat_prog.dc_entry
                };
                desc.callables_datas.push_back(dc_data);
            }
        }
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
    ImGui::Text("sample count: %d", m_optix_launch_params.sample_cnt + 1);
    ImGui::InputInt("max trace depth", &m_max_depth);
    m_max_depth = clamp(m_max_depth, 1, 128);
    if (m_optix_launch_params.config.max_depth != m_max_depth) {
        m_dirty = true;
    }

    if (ImGui::Checkbox("accumulate radiance", &m_accumulated_flag)) {
        m_dirty = true;
    }

    if (ImGui::Checkbox("path guiding", &m_use_path_guiding)) {
        m_optix_launch_params.config.use_path_guiding = m_use_path_guiding;

        // invalidate the model history
        size_t count = m_output_pixel_num * sizeof(vMF);
        CUDA_CHECK(cudaMemset(m_optix_launch_params.pre_model_buffer.GetDataPtr(), 0, count));
        CUDA_CHECK(cudaMemset(m_optix_launch_params.new_model_buffer.GetDataPtr(), 0, count));

        m_dirty = true;
    }
}
}// namespace Pupil::pt
