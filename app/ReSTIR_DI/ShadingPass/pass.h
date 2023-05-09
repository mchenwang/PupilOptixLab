#pragma once

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "util/timer.h"
#include "cuda/stream.h"
#include "optix/pass.h"
#include "optix/scene/camera.h"

#include "type.h"

struct ShadingPassSBTType : public Pupil::optix::EmptySBT {
};

class ShadingPass : public Pupil::Pass {
public:
    ShadingPass(std::string_view name = "ReSTIR DI Shading Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(Pupil::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    ShadingPassLaunchParams m_params;
    std::unique_ptr<Pupil::cuda::Stream> m_stream;
    std::unique_ptr<Pupil::optix::Pass<ShadingPassSBTType, ShadingPassLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;

    Pupil::Timer m_timer;
};