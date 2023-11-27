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
#include "system/profiler.h"
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

    Timer* timer;

    int frame_cnt = 0;
    struct CameraAnimation {
        int    frame_i;
        Float3 pos;
        Float3 ori;
    };

    // CameraAnimation ca[3] = {
    //     CameraAnimation{1, Float3(2, 2, 2), Float3(0.57, 0.57, 0.57)},
    //     CameraAnimation{300, Float3(17, 10, 8), Float3(0.75, 0.61, 0.21)},
    //     CameraAnimation{600, Float3(16, 11, -11), Float3(0.7, 0.6, -0.3)}};
    int ca_i = 1;

    std::vector<CameraAnimation> ca = {
        CameraAnimation(1, Float3(0.0324209, 1.52673, 4.88197), Float3(0.0076460172, -0.045055054, 0.9989554)),
        CameraAnimation(300, Float3(-0.062105987, 3.0363982, -0.09290689), Float3(-0.062105987, 3.0363982, -0.09290689) - Float3(0.018104862, -0.3029082, 0.9528476)),
        CameraAnimation(600, Float3(2.0159879, 3.036398, -0.15541534), Float3(2.0159879, 3.036398, -0.15541534) - Float3(0.36964566, -0.35970998, 0.85672057)),
        CameraAnimation(900, Float3(0.7926329, 4.578863, -3.5562882), Float3(0.7926329, 4.578863, -3.5562882) - Float3(0.7827882, -0.43416744, 0.44579676)),
        CameraAnimation(1200, Float3(-3.6460304, 6.6223083, -3.6487114), Float3(-3.6460304, 6.6223083, -3.6487114) - Float3(0.8263058, -0.4016645, -0.3948104)),
        CameraAnimation(1500, Float3(-2.7842517, 6.416301, -3.3986428), Float3(-2.7842517, 6.416301, -3.3986428) - Float3(-0.63194793, 0.4827861, -0.6062584)),
        CameraAnimation(1800, Float3(-2.3958085, 6.169997, -3.2289455), Float3(-2.3958085, 6.169997, -3.2289455) - Float3(-0.7551325, 0.5659387, -0.33087364))
        //    CameraAnimation(300, Float3(0.0071890475, 1.6754079, 1.585419), Float3(0.0076460172, -0.045055054, 0.9989554)),
        //    CameraAnimation(600, Float3(1.9829837, 1.7476434, 0.6422636), Float3(0.5549296, 0.041311685, 0.83087075)),
        //    CameraAnimation(900, Float3(1.982984, 1.7476435, 0.6422637), Float3(0.57397306, -0.4691987, 0.67112094)),
        //    CameraAnimation(1200, Float3(2.227737, 1.7938076, -0.3352181), Float3(0.95523596, -0.022378912, 0.2949903)),
        //    CameraAnimation(1500, Float3(2.3082454, 2.084975, -1.3941063), Float3(0.563943, -0.3425501, 0.75141335)),
        //    CameraAnimation(1800, Float3(2.4772153, 3.9605315, -3.3920057), Float3(0.9033614, -0.22635104, 0.3642742)),
        //    CameraAnimation(2100, Float3(-0.29726255, 4.582663, -3.905832), Float3(0.9733628, -0.20419423, 0.1042187)),
        //    CameraAnimation(2400, Float3(-0.29726246, 4.5826616, -3.905832), Float3(0.8366384, -0.10335392, -0.5379082)),
        //    CameraAnimation(2700, Float3(-0.29726273, 4.582662, -3.9058313), Float3(0.2379814, 0.16449077, -0.9572359)),
        //    CameraAnimation(3000, Float3(-0.2972626, 4.582661, -3.9058318), Float3(-0.5564907, 0.5463597, -0.6259412)),
        //    CameraAnimation(3300, Float3(-1.7502965, 5.525393, -4.384671), Float3(-0.65444994, 0.437079, -0.6169688))
    };
};

namespace Pupil::pt {
    PTPass::PTPass(std::string_view name) noexcept
        : Pupil::Pass(name), optix::Pass(sizeof(OptixLaunchParams)) {
        m_impl = new Impl();

        m_impl->timer = util::Singleton<Profiler>::instance()
                            ->AllocTimer(name, GetStream(), 60);

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
            m_impl->optix_launch_params.camera                  = m_impl->camera;
            m_impl->optix_launch_params.config.max_depth        = m_impl->max_depth;
            m_impl->optix_launch_params.config.accumulated_flag = m_impl->accumulated_flag;
            m_impl->optix_launch_params.sample_cnt              = 0;
            m_impl->optix_launch_params.random_seed             = 0;
            m_impl->frame_cnt                                   = 0;
            m_impl->ca_i                                        = 1;
            m_impl->optix_launch_params.handle                  = m_impl->scene->GetIASHandle(2, true);
            m_impl->optix_launch_params.emitters                = m_impl->scene->GetOptixEmitters();
            m_impl->dirty                                       = false;

            // auto m = m_impl->camera.camera_to_world;
            // Log::Info(" CameraAnimation(1, Float3({}, {}, {}), Float3({}, {}, {}))",
            //           m.r0.w, m.r1.w, m.r2.w, m.r0.z, m.r1.z, m.r2.z);
            // Log::Info(" Camera:\n {} {} {} {}\n {} {} {} {}\n {} {} {} {}",
            //           m.r0.x, m.r0.y, m.r0.z, m.r0.w,
            //           m.r1.x, m.r1.y, m.r1.z, m.r1.w,
            //           m.r2.x, m.r2.y, m.r2.z, m.r2.w,
            //           m.r0.x, m.r0.y, m.r0.z, m.r0.w);
        }

        m_impl->timer->Start();
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(m_impl->optix_launch_params_cuda_memory),
            &m_impl->optix_launch_params, sizeof(OptixLaunchParams),
            cudaMemcpyHostToDevice, *GetStream()));

        optix::Pass::Run(m_impl->optix_launch_params_cuda_memory,
                         m_impl->optix_launch_params.config.frame.width,
                         m_impl->optix_launch_params.config.frame.height);
        m_impl->timer->Stop();

        m_impl->frame_cnt++;
        // if (m_impl->frame_cnt <= m_impl->ca[m_impl->ca_i].frame_i) {
        //     float t   = ((float)(m_impl->frame_cnt - m_impl->ca[m_impl->ca_i - 1].frame_i) / (m_impl->ca[m_impl->ca_i].frame_i - m_impl->ca[m_impl->ca_i - 1].frame_i));
        //     auto  ori = Lerp(m_impl->ca[m_impl->ca_i - 1].ori, m_impl->ca[m_impl->ca_i].ori, Float3(t));
        //     auto  pos = Lerp(m_impl->ca[m_impl->ca_i - 1].pos, m_impl->ca[m_impl->ca_i].pos, Float3(t));

        //     auto to_world                                      = Pupil::MakeLookatToWorldMatrixRH(pos, ori, Float3(0, 1, 0));
        //     m_impl->optix_launch_params.camera.camera_to_world = Pupil::cuda::MakeMat4x4(to_world);
        //     m_impl->optix_launch_params.sample_cnt             = 0;
        //     m_impl->optix_launch_params.random_seed            = 0;

        //     // if (m_impl->frame_cnt % 30 == 0) {

        //     //     auto m = to_world;
        //     //     // Log::Info(" CameraAnimation(1, Float3({}, {}, {}), Float3({}, {}, {}))",
        //     //     //           m.r0.w, m.r1.w, m.r2.w, m.r0.z, m.r1.z, m.r2.z);
        //     //     // Log::Info(" Camera:\n {} {} {} {}\n {} {} {} {}\n {} {} {} {}",
        //     //     //           m.r0.x, m.r0.y, m.r0.z, m.r0.w,
        //     //     //           m.r1.x, m.r1.y, m.r1.z, m.r1.w,
        //     //     //           m.r2.x, m.r2.y, m.r2.z, m.r2.w,
        //     //     //           m.r0.x, m.r0.y, m.r0.z, m.r0.w);
        //     // }
        // } else {
        //     if (m_impl->ca_i < m_impl->ca.size() - 1) {
        //         m_impl->ca_i++;
        //     }
        // }
        // Synchronize();

        m_impl->optix_launch_params.sample_cnt += m_impl->optix_launch_params.config.accumulated_flag;
        ++m_impl->optix_launch_params.random_seed;
    }

    void PTPass::Synchronize() noexcept {
        m_stream->Synchronize();
    }

    void PTPass::Console() noexcept {
        Pupil::Pass::Console();

        util::Singleton<Profiler>::instance()->ShowPlot(name);

        ImGui::Text("sample cnt: %d", m_impl->frame_cnt);

        ImGui::InputInt("max trace depth", &m_impl->max_depth);
        m_impl->max_depth = clamp(m_impl->max_depth, 1, 128);
        if (m_impl->optix_launch_params.config.max_depth != m_impl->max_depth) {
            m_impl->dirty = true;
        }

        if (ImGui::Checkbox("accumulate radiance", &m_impl->accumulated_flag)) {
            m_impl->dirty = true;
        }

        if (ImGui::Button("reset")) {
            m_impl->dirty = true;
        }
    }

    void PTPass::BindingEventCallback() noexcept {
        auto event_center = util::Singleton<Event::Center>::instance();
        event_center->BindEvent(
            Event::DispatcherRender, Event::CameraChange,
            new Event::Handler0A([this]() {
                m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetToWorldMatrix());
                m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetSampleToCameraMatrix());

                m_impl->dirty = true;
            }));

        event_center->BindEvent(Event::DispatcherRender, Event::InstanceChange, new Event::Handler0A([this]() { m_impl->dirty = true; }));
        event_center->BindEvent(Event::DispatcherRender, Event::CameraChange, new Event::Handler0A([this]() { m_impl->frame_cnt = 0; }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::SceneReset,
            new Event::Handler0A([this]() {
                m_impl->scene = util::Singleton<World>::instance()->GetScene();

                m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetToWorldMatrix());
                m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetSampleToCameraMatrix());

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
                    // m_impl->optix_launch_params.frame_buffer.SetData(buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME)->cuda_ptr, m_impl->output_pixel_num);

                    BufferDesc desc{
                        .name           = "pt accum buffer",
                        .flag           = EBufferFlag::None,
                        .width          = static_cast<uint32_t>(m_impl->scene->film_w),
                        .height         = static_cast<uint32_t>(m_impl->scene->film_h),
                        .stride_in_byte = sizeof(float) * 4};
                    m_impl->optix_launch_params.accum_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name = "pt result";
                    desc.flag = EBufferFlag::AllowDisplay;
                    m_impl->optix_launch_params.frame_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

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