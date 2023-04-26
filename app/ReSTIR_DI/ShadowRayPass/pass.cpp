#include "pass.h"
#include "imgui.h"

#include "../GBufferPass/pass.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "scene/scene.h"
#include "optix/scene/scene.h"

#include "util/event.h"
#include "util/util.h"
#include "system/system.h"
#include "system/gui.h"
#include "system/resource.h"

using namespace Pupil;

extern "C" char g_restir_di_shadow_ptx[];

namespace {
double m_time_cost = 0.f;
bool m_flag = true;
}// namespace

ShadowRayPass::ShadowRayPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<ShadowRayPassSBTType, ShadowRayPassLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void ShadowRayPass::Run() noexcept {
    m_timer.Start();
    if (m_flag) {
        m_optix_pass->Run(m_params, m_params.frame.width, m_params.frame.height);
        m_optix_pass->Synchronize();
    }
    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void ShadowRayPass::SetScene(Pupil::World *world) noexcept {
    m_params.frame.width = world->scene->sensor.film.w;
    m_params.frame.height = world->scene->sensor.film.h;
    m_params.handle = world->optix_scene->ias_handle;

    m_output_pixel_num = m_params.frame.width * m_params.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    auto pos_buf = buf_mngr->GetBuffer(GBufferPass::POS);
    auto nor_buf = buf_mngr->GetBuffer(GBufferPass::NORMAL);
    auto alb_buf = buf_mngr->GetBuffer(GBufferPass::ALBEDO);

    m_params.position.SetData(pos_buf->cuda_res.ptr, m_output_pixel_num);
    m_params.normal.SetData(nor_buf->cuda_res.ptr, m_output_pixel_num);
    m_params.albedo.SetData(alb_buf->cuda_res.ptr, m_output_pixel_num);

    auto reservoir_buf = buf_mngr->GetBuffer("final screen reservoir");
    m_params.reservoirs.SetData(reservoir_buf->cuda_res.ptr, m_output_pixel_num);

    {
        optix::SBTDesc<ShadowRayPassSBTType> desc{};
        desc.ray_gen_data = {
            .program_name = "__raygen__main",
            .data = ShadowRayPassSBTType::RayGenDataType{}
        };
        {
            using HitGroupDataRecord = decltype(desc)::Pair<ShadowRayPassSBTType::HitGroupDataType>;
            for (auto &&shape : world->scene->shapes) {
                HitGroupDataRecord hit_default_data{};
                hit_default_data.program_name = "__closesthit__default";

                desc.hit_datas.push_back(hit_default_data);
                desc.hit_datas.push_back(hit_default_data);
            }
        }
        {
            decltype(desc)::Pair<ShadowRayPassSBTType::MissDataType> miss_data = {
                .program_name = "__miss__default",
                .data = ShadowRayPassSBTType::MissDataType{}
            };
            desc.miss_datas.push_back(miss_data);
            desc.miss_datas.push_back(miss_data);
        }
        m_optix_pass->InitSBT(desc);
    }
}

void ShadowRayPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto rt_module = module_mngr->GetModule(g_restir_di_shadow_ptx);
    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::ProgramDesc desc{
            .module_ptr = rt_module,
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::ProgramDesc desc{
            .module_ptr = rt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void ShadowRayPass::BindingEventCallback() noexcept {
    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((World *)p);
    });
}

void ShadowRayPass::Inspector() noexcept {
    ImGui::Text("cost: %d ms", (int)m_time_cost);
    ImGui::Checkbox("use shadow", &m_flag);
}
