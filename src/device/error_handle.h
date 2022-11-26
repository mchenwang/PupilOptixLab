#pragma once

#include <optix.h>
#include <cuda_runtime.h>
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
inline void OptixCheck(OptixResult res, const char *call, const char *file, unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::wstringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        std::wcerr << ss.str().c_str();
        assert(false);
    }
}
inline void OptixCheckLog(
    OptixResult res,
    const char *log,
    size_t sizeof_log,
    size_t sizeof_log_returned,
    const char *call,
    const char *file,
    unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::wstringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
        std::wcerr << ss.str().c_str();
        assert(false);
    }
}
#define CUDA_CHECK(call) CudaCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK(call) OptixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                  \
    do {                                                       \
        char LOG[400];                                         \
        size_t LOG_SIZE = sizeof(LOG);                         \
        OptixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, \
                      __FILE__, __LINE__);                     \
    } while (false)

#define CUDA_FREE(var)                                           \
    do {                                                         \
        if (var)                                                 \
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(var))); \
    } while (false)
