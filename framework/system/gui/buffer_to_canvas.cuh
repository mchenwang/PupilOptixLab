#pragma once

#include "cuda/stream.h"
#include <cuda.h>

namespace Pupil {
    void CopyFromFloat1(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept;
    void CopyFromFloat2(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept;
    void CopyFromFloat3(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept;
}// namespace Pupil