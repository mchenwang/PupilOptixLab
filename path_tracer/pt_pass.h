#pragma once

#include "type.h"

#include "system/pass.h"
#include "system/resource.h"
#include "scene/scene.h"
#include "optix/pass.h"
#include "optix/scene/scene.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil::pt {
struct SBTTypes {
    using RayGenDataType = Pupil::pt::RayGenData;
    using MissDataType = Pupil::pt::MissData;
    using HitGroupDataType = Pupil::pt::HitGroupData;
};

class PTPass : public Pass {
public:
    PTPass(std::string_view name = "Path Tracing") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;
    virtual void SetScene(scene::Scene *) noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;
    void SetSBT(scene::Scene *) noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    std::unique_ptr<optix::Scene> m_optix_scene;
    size_t m_output_pixel_num = 0;
    CUdeviceptr m_accum_buffer = 0;

    std::atomic_bool m_dirty = true;

    Timer m_timer;
};
}// namespace Pupil::pt