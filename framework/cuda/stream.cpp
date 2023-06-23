#include "stream.h"

#include "util.h"

namespace Pupil::cuda {
Stream::Stream() noexcept {
    // CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    //CUDA_CHECK(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
}
Stream::~Stream() noexcept {
    CUDA_CHECK(cudaStreamDestroy(m_stream));
    //CUDA_CHECK(cudaEventDestroy(m_event));
}

void Stream::Synchronize() noexcept { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }
}// namespace Pupil::cuda