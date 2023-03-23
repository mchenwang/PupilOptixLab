#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "vec_math.h"

#include <sstream>
#include <assert.h>
#include <iostream>

inline void CudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        std::wstringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        std::wcerr << ss.str().c_str();
        assert(false);
    }
}
inline void CudaCheck(CUresult error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        std::wstringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << error << "' (" << file << ":" << line << ")\n";

        std::wcerr << ss.str().c_str();
        assert(false);
    }
}
inline void CudaSyncCheck(const char *file, unsigned int line) {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
           << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";

        std::wcerr << ss.str().c_str();
        assert(false);
    }
}

#define CUDA_CHECK(call) CudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() CudaSyncCheck(__FILE__, __LINE__)

#define CUDA_FREE(var)                                           \
    do {                                                         \
        if (var)                                                 \
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(var))); \
        var = 0;                                                 \
    } while (false)

namespace Pupil::cuda {
inline CUdeviceptr CudaMemcpyToDevice(void *src, size_t size) {
    CUdeviceptr device_memory = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_memory), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void **>(device_memory), src, size, cudaMemcpyHostToDevice));
    return device_memory;
}
inline void CudaMemcpyToDevice(CUdeviceptr dst, void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void **>(dst), src, size, cudaMemcpyHostToDevice));
}

inline void CudaMemcpyToHost(void *dst, CUdeviceptr src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, reinterpret_cast<const void *>(src), size, cudaMemcpyDeviceToHost));
}
inline void CudaMemcpyToHost(void *dst, const void *src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}
}// namespace Pupil::cuda