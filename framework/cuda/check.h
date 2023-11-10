
#include "util/log.h"

#include <cuda.h>
#include <cuda_runtime.h>

inline void CudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line) {
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA call({}) failed with error: {}", call, cudaGetErrorString(error));
        Pupil::Log::Error("location:{} : {}.", file, line);
        assert(false);
    }
}
inline void CudaCheck(CUresult error, const char* call, const char* file, unsigned int line) {
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA call({}) failed with error: {}", call, error);
        Pupil::Log::Error("location:{} : {}.", file, line);
        assert(false);
    }
}
inline void CudaSyncCheck(const char* file, unsigned int line) {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA error on synchronize with error: {}", cudaGetErrorString(error));
        Pupil::Log::Error("location:{} : {}.", file, line);
        assert(false);
    }
}

#define CUDA_CHECK(call) CudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() CudaSyncCheck(__FILE__, __LINE__)

#define CUDA_FREE(var)                                          \
    do {                                                        \
        if (var)                                                \
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(var))); \
        var = 0;                                                \
    } while (false)

#define CUDA_FREE_ASYNC(var, stream)                                         \
    do {                                                                     \
        if (var)                                                             \
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(var), stream)); \
        var = 0;                                                             \
    } while (false)
