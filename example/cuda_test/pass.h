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
    Pupil::util::CountableRef<Pupil::cuda::Stream> m_stream    = nullptr;
    unsigned int                                   m_frame_cnt = 0;
    Pupil::cuda::RWArrayView<float4>               m_output;
};