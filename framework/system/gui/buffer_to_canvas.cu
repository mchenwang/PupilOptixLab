#include "buffer_to_canvas.cuh"
#include "cuda/kernel.h"

namespace Pupil {
    void CopyFromFloat1(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel1D(
            size, [dst, src] __device__(unsigned int index, unsigned int size) {
                auto output   = reinterpret_cast<float4*>(dst);
                auto input    = reinterpret_cast<float*>(src);
                output[index] = make_float4(input[index], input[index], input[index], 1.f);
            },
            stream);
    }

    void CopyFromFloat2(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel1D(
            size, [dst, src] __device__(unsigned int index, unsigned int size) {
                auto output   = reinterpret_cast<float4*>(dst);
                auto input    = reinterpret_cast<float2*>(src);
                output[index] = make_float4(input[index].x, input[index].y, 0.f, 1.f);
            },
            stream);
    }

    void CopyFromFloat3(CUdeviceptr dst, CUdeviceptr src, unsigned int size, cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel1D(
            size, [dst, src] __device__(unsigned int index, unsigned int size) {
                auto output   = reinterpret_cast<float4*>(dst);
                auto input    = reinterpret_cast<float3*>(src);
                output[index] = make_float4(input[index].x, input[index].y, input[index].z, 1.f);
            },
            stream);
    }
}// namespace Pupil