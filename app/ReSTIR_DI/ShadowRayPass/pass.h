#pragma once

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "util/timer.h"
#include "cuda/stream.h"
#include "optix/pass.h"
#include "optix/scene/camera.h"

#include "type.h"

struct ShadowRayPassSBTType {
    using RayGenDataType = ShadowRayPassRayGenData;
    using MissDataType = ShadowRayPassMissData;
    using HitGroupDataType = ShadowRayPassHitGroupData;
};

class ShadowRayPass : public Pupil::Pass {
public:
    ShadowRayPass(std::string_view name = "ReSTIR DI ShadowRay Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(Pupil::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    ShadowRayPassLaunchParams m_params;
    std::unique_ptr<Pupil::cuda::Stream> m_stream;
    std::unique_ptr<Pupil::optix::Pass<ShadowRayPassSBTType, ShadowRayPassLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;

    Pupil::Timer m_timer;
};