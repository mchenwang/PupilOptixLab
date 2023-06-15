#pragma once

#include "type.h"

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "scene/scene.h"
#include "optix/pass.h"
#include "optix/scene/scene.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil::pt {
struct SBTTypes : public optix::EmptySBT {
    using HitGroupDataType = Pupil::pt::HitGroupData;
};

class PTPass : public Pass {
public:
    explicit PTPass(std::string_view name = "Path Tracing") noexcept;
    void Run() noexcept override;
    void Inspector() noexcept override;

    void BeforeRunning() noexcept override {}
    void AfterRunning() noexcept override {}

    void SetScene(World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;
    void SetSBT(scene::Scene *) noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;
    Buffer *m_accum_buffer = nullptr;

    std::atomic_bool m_dirty = true;
    CameraHelper *m_world_camera = nullptr;

    Timer m_timer;
};
}// namespace Pupil::pt
