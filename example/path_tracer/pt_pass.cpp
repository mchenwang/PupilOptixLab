#include "pt_pass.h"
#include "type.h"
#include "imgui.h"

#include "cuda/context.h"
#include "cuda/check.h"
#include "optix/context.h"
#include "optix/pipeline.h"

#include "system/system.h"
#include "system/event.h"
#include "system/buffer.h"
#include "system/world.h"
#include "system/gui/pass.h"

#include "render/camera.h"

#include <memory>
#include <mutex>

extern "C" char embedded_ptx_code[];

namespace Pupil {
    extern uint32_t g_window_w;
    extern uint32_t g_window_h;
}// namespace Pupil

struct Pupil::pt::PTPass::Impl {
    int  max_depth;
    bool accumulated_flag;

    Pupil::Scene*        scene = nullptr;
    Pupil::optix::Camera camera;

    util::CountableRef<cuda::Stream> stream;

    OptixLaunchParams optix_launch_params;
    CUdeviceptr       optix_launch_params_cuda_memory = 0;

    size_t output_pixel_num = 0;

    std::atomic_bool dirty = true;
};

namespace Pupil::pt {
    PTPass::PTPass(std::string_view name) noexcept
        : Pupil::Pass(name), optix::Pass(sizeof(OptixLaunchParams)) {
        m_impl = new Impl();

        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&m_impl->optix_launch_params_cuda_memory),
            sizeof(OptixLaunchParams),
            *GetStream()));

        InitPipeline();
        BindingEventCallback();
    }

    PTPass::~PTPass() noexcept {
        CUDA_FREE(m_impl->optix_launch_params_cuda_memory);
        delete m_impl;
    }

    void PTPass::OnRun() noexcept {
        if (m_impl->dirty) {
            m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera().GetToWorldMatrix());
            m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera().GetSampleToCameraMatrix());

            m_impl->optix_launch_params.camera                  = m_impl->camera;
            m_impl->optix_launch_params.config.max_depth        = m_impl->max_depth;
            m_impl->optix_launch_params.config.accumulated_flag = m_impl->accumulated_flag;
            m_impl->optix_launch_params.sample_cnt              = 0;
            m_impl->optix_launch_params.random_seed             = 0;
            m_impl->optix_launch_params.handle                  = m_impl->scene->GetIASHandle(2, true);
            m_impl->optix_launch_params.emitters                = m_impl->scene->GetOptixEmitters();
            m_impl->dirty                                       = false;
        }

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(m_impl->optix_launch_params_cuda_memory),
            &m_impl->optix_launch_params, sizeof(OptixLaunchParams),
            cudaMemcpyHostToDevice, *GetStream()));

        optix::Pass::Run(m_impl->optix_launch_params_cuda_memory,
                         m_impl->optix_launch_params.config.frame.width,
                         m_impl->optix_launch_params.config.frame.height);
        Synchronize();

        m_impl->optix_launch_params.sample_cnt += m_impl->optix_launch_params.config.accumulated_flag;
        ++m_impl->optix_launch_params.random_seed;
    }

    void PTPass::Console() noexcept {
        Pupil::Pass::Console();
        ImGui::InputInt("max trace depth", &m_impl->max_depth);
        m_impl->max_depth = clamp(m_impl->max_depth, 1, 128);
        if (m_impl->optix_launch_params.config.max_depth != m_impl->max_depth) {
            m_impl->dirty = true;
        }

        if (ImGui::Checkbox("accumulate radiance", &m_impl->accumulated_flag)) {
            m_impl->dirty = true;
        }
    }

    void PTPass::BindingEventCallback() noexcept {
        auto event_center = util::Singleton<Event::Center>::instance();
        event_center->BindEvent(
            Event::DispatcherRender, Event::CameraChange,
            new Event::Handler0A([this]() {
                m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera().GetToWorldMatrix());
                m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera().GetSampleToCameraMatrix());

                m_impl->dirty = true;
            }));

        event_center->BindEvent(Event::DispatcherRender, Event::InstanceChange, new Event::Handler0A([this]() { m_impl->dirty = true; }));
        event_center->BindEvent(
            Event::DispatcherRender, Event::SceneReset,
            new Event::Handler0A([this]() {
                m_impl->scene = util::Singleton<World>::instance()->GetScene();

                m_impl->optix_launch_params.config.frame.width      = m_impl->scene->film_w;
                m_impl->optix_launch_params.config.frame.height     = m_impl->scene->film_h;
                m_impl->optix_launch_params.config.max_depth        = m_impl->scene->max_depth;
                m_impl->optix_launch_params.config.accumulated_flag = true;

                m_impl->max_depth        = m_impl->optix_launch_params.config.max_depth;
                m_impl->accumulated_flag = m_impl->optix_launch_params.config.accumulated_flag;

                m_impl->optix_launch_params.random_seed = 0;
                m_impl->optix_launch_params.sample_cnt  = 0;

                m_impl->output_pixel_num = m_impl->optix_launch_params.config.frame.width *
                                           m_impl->optix_launch_params.config.frame.height;
                auto buf_mngr = util::Singleton<BufferManager>::instance();
                {
                    m_impl->optix_launch_params.frame_buffer.SetData(buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME)->cuda_ptr, m_impl->output_pixel_num);

                    BufferDesc desc{
                        .name           = "pt accum buffer",
                        .flag           = EBufferFlag::None,
                        .width          = static_cast<uint32_t>(m_impl->scene->film_w),
                        .height         = static_cast<uint32_t>(m_impl->scene->film_h),
                        .stride_in_byte = sizeof(float) * 4};
                    m_impl->optix_launch_params.accum_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "albedo";
                    desc.flag           = EBufferFlag::AllowDisplay;
                    desc.stride_in_byte = sizeof(float) * 4;
                    m_impl->optix_launch_params.albedo_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name = "normal";
                    m_impl->optix_launch_params.normal_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);
                }

                m_impl->optix_launch_params.handle = m_impl->scene->GetIASHandle(2, true);

                {
                    auto instances    = m_impl->scene->GetInstances();
                    auto instance_num = static_cast<unsigned int>(instances.size());
                    m_sbt->SetRayGenRecord<void>();
                    m_sbt->SetHitgroupRecord<HitGroupData>(instance_num * 2);
                    m_sbt->SetMissRecord<void>(2);

                    m_sbt->BindData("ray gen", nullptr);
                    m_sbt->BindData("miss", nullptr, 0);
                    m_sbt->BindData("miss shadow", nullptr, 1);

                    for (auto i = 0u; i < instance_num; i++) {
                        HitGroupData hit;
                        hit.geo = instances[i].shape->GetOptixGeometry();
                        hit.mat = instances[i].material->GetOptixMaterial();

                        if (instances[i].emitter != nullptr) {
                            hit.emitter_index = m_impl->scene->GetEmitterIndex(instances[i].emitter);
                        } else
                            hit.emitter_index = -1;

                        if (hit.geo.type == optix::Geometry::EType::TriMesh) {
                            m_sbt->BindData("hit", &hit, i * 2);
                            m_sbt->BindData("hit shadow", &hit, i * 2 + 1);
                        } else if (hit.geo.type == optix::Geometry::EType::Sphere) {
                            m_sbt->BindData("hit sphere", &hit, i * 2);
                            m_sbt->BindData("hit shadow sphere", &hit, i * 2 + 1);
                        } else {
                            m_sbt->BindData("hit curve", &hit, i * 2);
                            m_sbt->BindData("hit shadow curve", &hit, i * 2 + 1);
                        }
                    }

                    m_sbt->Finish();
                }

                m_impl->dirty = true;
            }));
    }

    void PTPass::InitPipeline() noexcept {
        m_pipeline->SetPipelineLaunchParamsVariableName("optix_launch_params");
        m_pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Sphere);
        m_pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Curve);

        auto pt_module     = m_pipeline->CreateModule(optix::EModuleType::UserDefined, embedded_ptx_code);
        auto sphere_module = m_pipeline->CreateModule(optix::EModuleType::BuiltinSphereIS);
        auto curve_module  = m_pipeline->CreateModule(optix::EModuleType::BuiltinCurveIS);

        m_pipeline->CreateRayGen("ray gen").SetModule(pt_module).SetEntry("__raygen__main");
        m_pipeline->CreateMiss("miss").SetModule(pt_module).SetEntry("__miss__default");
        m_pipeline->CreateHitgroup("hit").SetCHModule(pt_module).SetCHEntry("__closesthit__default");
        m_pipeline->CreateHitgroup("hit sphere").SetCHModule(pt_module).SetCHEntry("__closesthit__default").SetISModule(sphere_module);
        m_pipeline->CreateHitgroup("hit curve").SetCHModule(pt_module).SetCHEntry("__closesthit__default").SetISModule(curve_module);
        m_pipeline->CreateMiss("miss shadow").SetModule(pt_module).SetEntry("__miss__shadow");
        m_pipeline->CreateHitgroup("hit shadow").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow");
        m_pipeline->CreateHitgroup("hit shadow sphere").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow").SetISModule(sphere_module);
        m_pipeline->CreateHitgroup("hit shadow curve").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow").SetISModule(curve_module);

        m_pipeline->Finish();
    }

}// namespace Pupil::pt