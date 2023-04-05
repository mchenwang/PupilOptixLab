#pragma once

#include "type.h"

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "scene/scene.h"
#include "optix/pass.h"
#include "optix/scene/scene.h"
#include "optix/scene/camera.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil::ddgi::gbuffer {
struct SBTTypes {
    using RayGenDataType = Pupil::ddgi::gbuffer::RayGenData;
    using MissDataType = Pupil::ddgi::gbuffer::MissData;
    using HitGroupDataType = Pupil::ddgi::gbuffer::HitGroupData;
};

class GBufferPass : public Pass {
public:
    GBufferPass(std::string_view name = "DDGI GBuffer Pass") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;
    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;
    void SetSBT(scene::Scene *) noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;
    Buffer *m_albedo = nullptr;
    Buffer *m_normal = nullptr;

    std::atomic_bool m_dirty = true;
    optix::CameraHelper *m_world_camera = nullptr;

    Timer m_timer;
};
}// namespace Pupil::ddgi::gbuffer