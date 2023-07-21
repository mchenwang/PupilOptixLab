#pragma once

#include "system/pass.h"
#include "system/world.h"
#include "util/timer.h"

#include "cuda/stream.h"

class CudaPass : public Pupil::Pass {
public:
    CudaPass(std::string_view name = "Cuda") noexcept;
    virtual void OnRun() noexcept override;

private:
    std::unique_ptr<Pupil::cuda::Stream> m_stream = nullptr;
};