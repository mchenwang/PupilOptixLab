#pragma once

#include "type.h"

#include "system/pass.h"
#include "system/buffer.h"
#include "world/world.h"
#include "resource/scene.h"
#include "optix/pass.h"

#include "cuda/stream.h"

#include <memory>
#include <mutex>

namespace Pupil::pt {
struct SBTTypes : public optix::EmptySBT {
    using HitGroupDataType = Pupil::pt::HitGroupData;
};

class PTPass : public Pass {
public:
    PTPass(std::string_view name = "Path Tracing") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;

    void SetScene(world::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;
    void SetSBT(resource::Scene *) noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;

    std::atomic_bool m_dirty = true;
    world::CameraHelper *m_world_camera = nullptr;
};
}// namespace Pupil::pt