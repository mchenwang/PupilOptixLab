#pragma once

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "util/timer.h"
#include "cuda/stream.h"
#include "optix/pass.h"

#include "type.h"

struct SpatialReusePassSBTType : public Pupil::optix::EmptySBT {
};

class SpatialReusePass : public Pupil::Pass {
public:
    SpatialReusePass(std::string_view name = "ReSTIR DI Spatial Reuse Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(Pupil::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    SpatialReusePassLaunchParams m_params;
    std::unique_ptr<Pupil::cuda::Stream> m_stream;
    std::unique_ptr<Pupil::optix::Pass<SpatialReusePassSBTType, SpatialReusePassLaunchParams>> m_optix_pass;

    std::atomic_bool m_dirty = true;
    Pupil::util::Camera *m_camera;
    unsigned int m_output_pixel_num = 0;

    Pupil::Timer m_timer;
};