#include "stream.h"

namespace Pupil::cuda {
Stream::Stream() noexcept {
    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
}
Stream::~Stream() noexcept {
    CUDA_CHECK(cudaStreamDestroy(m_stream));
    CUDA_CHECK(cudaEventDestroy(m_event));
}

}// namespace Pupil::cuda