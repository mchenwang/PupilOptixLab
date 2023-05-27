#pragma once

#include "system/pass.h"
#include "system/world.h"
#include "util/timer.h"

#include "cuda/stream.h"

class CudaPass : public Pupil::Pass {
public:
    CudaPass(std::string_view name = "Cuda") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override {}

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

private:
    Pupil::Timer m_timer;

    std::unique_ptr<Pupil::cuda::Stream> m_stream = nullptr;
    unsigned int m_frame_cnt = 0;
};