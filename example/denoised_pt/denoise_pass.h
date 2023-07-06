#pragma once

#include "system/pass.h"
#include "system/resource.h"
#include "system/world.h"
#include "scene/scene.h"
#include "optix/denoiser.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil {
class DenoisePass : public Pass {
public:
    static bool s_enabled_flag;

    DenoisePass(std::string_view name = "Denoise") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(World *) noexcept;

private:
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Denoiser> m_denoiser;

    Timer m_timer;
};
}// namespace Pupil