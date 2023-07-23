#pragma once

#include "cuda/stream.h"
#include "cuda/data_view.h"
#include <cuda.h>

namespace Pupil {
void CopyFloat1BufferToCanvas(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream *stream) noexcept;
void CopyFloat2BufferToCanvas(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream *stream) noexcept;
void CopyFloat3BufferToCanvas(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream *stream) noexcept;
}