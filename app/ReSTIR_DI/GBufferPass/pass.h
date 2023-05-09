#pragma once

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "util/timer.h"
#include "cuda/stream.h"
#include "optix/pass.h"
#include "optix/scene/camera.h"

#include "type.h"

struct GBufferPassSBTType : public Pupil::optix::EmptySBT {
    using HitGroupDataType = GBufferPassHitGroupData;
};

class GBufferPass : public Pupil::Pass {
public:
    static constexpr std::string_view POS = "gbuffer position";
    static constexpr std::string_view NORMAL = "gbuffer normal";
    static constexpr std::string_view ALBEDO = "gbuffer albedo";

    GBufferPass(std::string_view name = "ReSTIR DI GBuffer Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(Pupil::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    GBufferPassLaunchParams m_params;
    std::unique_ptr<Pupil::cuda::Stream> m_stream;
    std::unique_ptr<Pupil::optix::Pass<GBufferPassSBTType, GBufferPassLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;
    Pupil::Buffer *m_pos_buf = nullptr;
    Pupil::Buffer *m_nor_buf = nullptr;
    Pupil::Buffer *m_alb_buf = nullptr;

    std::atomic_bool m_dirty = true;
    Pupil::CameraHelper *m_world_camera = nullptr;

    Pupil::Timer m_timer;
};