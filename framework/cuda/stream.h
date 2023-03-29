#pragma once

#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Pupil::cuda {
class Stream {
public:
    Stream() noexcept;
    ~Stream() noexcept;

    operator cudaStream_t() const noexcept { return m_stream; }

    [[nodiscard]] bool IsAvailable() noexcept { return cudaSuccess == cudaEventQuery(m_event); }

    void Synchronize() noexcept { CUDA_CHECK(cudaEventSynchronize(m_event)); }
    void Signal() noexcept { CUDA_CHECK(cudaEventRecord(m_event, m_stream)); }

private:
    cudaStream_t m_stream;
    cudaEvent_t m_event;
};
}// namespace Pupil::cuda