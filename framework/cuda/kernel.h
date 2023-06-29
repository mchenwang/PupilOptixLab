#pragma once

#include "preprocessor.h"
#include "stream.h"

#ifdef PUPIL_CUDA
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

namespace Pupil::cuda {
namespace detail {
template<typename Func>
__global__ void Execute(Func kernel) {
    kernel();
}

template<typename Func>
__global__ void Execute(unsigned int task_size, Func kernel) {
    unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (task_id >= task_size) return;
    kernel(task_id, task_size);
}

template<typename Func>
__global__ void Execute(uint2 task_size, Func kernel) {
    uint2 task_id;
    task_id.x = threadIdx.x + blockIdx.x * blockDim.x;
    task_id.y = threadIdx.y + blockIdx.y * blockDim.y;

    if (task_id.x >= task_size.x) return;
    if (task_id.y >= task_size.y) return;
    kernel(task_id, task_size);
}
}// namespace detail

template<typename Func>
void LaunchKernel(Func kernel, Stream *stream) {
    detail::Execute<<<1, 1, 0, *stream>>>(kernel);
}

template<typename Func>
void LaunchKernel1D(unsigned int task_size, Func kernel, Stream *stream) {
    static int block_size = 32;
    int grid_size = (task_size + block_size - 1) / block_size;
    detail::Execute<<<grid_size, block_size, 0, *stream>>>(task_size, kernel);
}

template<typename Func>
void LaunchKernel2D(uint2 task_size, Func kernel, Stream *stream) {
    int block_size_x = 32;
    int block_size_y = 32;
    int grid_size_x = (task_size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (task_size.y + block_size_y - 1) / block_size_y;
    detail::Execute<<<dim3(grid_size_x, grid_size_y),
                      dim3(block_size_x, block_size_y),
                      0, *stream>>>(task_size, kernel);
}
}// namespace Pupil::cuda
#endif