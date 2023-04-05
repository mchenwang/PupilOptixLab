#include "pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui.h"

// 构建期通过CMake将.cu代码编译并嵌入到.c文件中，
// 代码指令由char类型保存，只需要声明extern即可获取
extern "C" char ddgi_pt_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
int m_max_depth;
int m_spp = 1;
double m_time_cnt = 1.;

int m_show_type = 0;
}// namespace

namespace Pupil::ddgi::pt {
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
        // 由于ui线程和渲染线程分离，所以在渲染前先检查是否通过ui修改了渲染参数
        if (m_dirty) {
            m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());
            m_optix_launch_params.config.max_depth = m_max_depth;
            m_optix_launch_params.random_seed = 0;
            m_optix_launch_params.spp = m_spp;
            m_dirty = false;
        }

        // 获取用于在gui上展示的的Buffer资源，每次渲染开始时都不一样(使用了flip model，一共有两个buffer，来回切换)
        // 只要调用GetCurrentRenderOutputBuffer即可获得，
        // 该buffer是cuda与dx12的共享资源，所以叫shared_buffer
        auto &frame_buffer =
            util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;

        if (m_show_type == 0) {// pt
            // frame_buffer写入的内容将会被展示到gui上
            m_optix_launch_params.frame_buffer.SetData(
                frame_buffer.cuda_ptr, m_output_pixel_num);
            m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                              m_optix_launch_params.config.frame.height);
            m_optix_pass->Synchronize();// 等待optix渲染结束

            ++m_optix_launch_params.random_seed;
        } else if (m_show_type == 1) {// albedo
            auto buf_mngr = util::Singleton<BufferManager>::instance();
            auto albedo = buf_mngr->GetBuffer("ddgi_albedo");
            cudaMemcpyAsync((void *)frame_buffer.cuda_ptr, (void *)albedo->cuda_res.ptr,
                            m_output_pixel_num * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream());
            cudaStreamSynchronize(m_stream->GetStream());
        } else if (m_show_type == 2) {// normal
            auto buf_mngr = util::Singleton<BufferManager>::instance();
            auto normal = buf_mngr->GetBuffer("ddgi_normal");
            cudaMemcpyAsync((void *)frame_buffer.cuda_ptr, (void *)normal->cuda_res.ptr,
                            m_output_pixel_num * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream());
            cudaStreamSynchronize(m_stream->GetStream());
        }
    }
    m_timer.Stop();
    m_time_cnt = m_timer.ElapsedMilliseconds();
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule(ddgi_pt_pass_embedded_ptx_code);

    // 创建optix pipeline，需要填写.cu中对应的函数入口，
    // 产生光线(每一个像素都会执行，相当于一个线程)：命名前缀必须是__raygen__
    // 光线击中(当optixTrace()发出的光线与场景相交(最近的交点)时会执行)：命名前缀必须是__closesthit__
    // 光线未击中：命名前缀必须是__miss__
    // 光追算法需要追踪view ray和shadow ray(判断是否在阴影中，即交点与光源中间是否有遮挡)
    // 因此对closesthit和miss需要有default和shadow两种处理，分别对应两个函数的入口
    // 但在gbuffer渲染时不需要shadow ray
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
    // mesh的求交使用built-in的三角形求交模块（默认）
    // 球的求交使用built-in的球求交模块(这里使用module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE)生成)
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
    // 对于场景初始化参数
    m_world_camera = world->optix_scene->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;

    m_max_depth = m_optix_launch_params.config.max_depth;
    m_spp = 1;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.spp = m_spp;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;

    m_optix_launch_params.frame_buffer.SetData(0, 0);
    m_optix_launch_params.handle = world->optix_scene->ias_handle;
    m_optix_launch_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    SetSBT(world->scene.get());

    m_dirty = true;
}

void PTPass::SetSBT(scene::Scene *scene) noexcept {
    // 将场景数据绑定到sbt中
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
    // 相机参数改变后，需要在渲染前将改变后的参数传入optix的pipeline中，
    // 由于是多线程异步实现的，需要m_dirty原子操作标记一下
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    // 导入新的场景时，需要重新执行SetScene的逻辑
    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void PTPass::Inspector() noexcept {
    constexpr auto show_type = std::array{ "pt result", "albedo", "normal" };
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
    }
    ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", m_time_cnt, 1000.0f / m_time_cnt);
}
}// namespace Pupil::ddgi::pt