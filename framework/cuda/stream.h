#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace Pupil::cuda {
// one task corresponds to one stream, so the event is not necessary
class Stream {
public:
    Stream() noexcept;
    ~Stream() noexcept;

    operator cudaStream_t() const noexcept { return m_stream; }
    [[nodiscard]] cudaStream_t GetStream() const noexcept { return m_stream; }
    //[[nodiscard]] cudaEvent_t GetEvent() const noexcept { return m_event; }
    //[[nodiscard]] bool IsAvailable() noexcept { return cudaSuccess == cudaEventQuery(m_event); }

    void Synchronize() noexcept;
    //void Synchronize() noexcept { CUDA_CHECK(cudaEventSynchronize(m_event)); }
    //void Signal() noexcept { CUDA_CHECK(cudaEventRecord(m_event, m_stream)); }

private:
    cudaStream_t m_stream;
    //cudaEvent_t m_event;
};
}// namespace Pupil::cuda