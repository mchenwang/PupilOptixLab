#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "vec_math.h"

#ifndef PUPIL_OPTIX
#include "util/log.h"

#include <sstream>
#include <assert.h>
#include <iostream>
#include <unordered_map>

inline void CudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA call({}) failed with error: {}\n\tlocation:{} : {}.\n", call, cudaGetErrorString(error), file, line);
        assert(false);
    }
}
inline void CudaCheck(CUresult error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA call({}) failed with error: {}\n\tlocation:{} : {}.\n", call, error, file, line);
        assert(false);
    }
}
inline void CudaSyncCheck(const char *file, unsigned int line) {
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        Pupil::Log::Error("CUDA error on synchronize with error: {}\n\tlocation:{} : {}.\n", cudaGetErrorString(error), file, line);
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
#endif

namespace Pupil::cuda {
template<typename T>
class DynamicArray {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);
    CUdeviceptr m_num CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE DynamicArray() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data, CUdeviceptr num) noexcept {
        m_data = cuda_data;
        m_num = num;
    }
    CUDA_HOSTDEVICE T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE T &operator[](unsigned int index) const noexcept {
        return *reinterpret_cast<T *>(m_data + index * sizeof(T));
    }

#ifdef PUPIL_CPP
    CUDA_HOSTDEVICE CUdeviceptr GetNum() const noexcept { return m_num; }
#else
    CUDA_DEVICE void Clear() noexcept { *reinterpret_cast<unsigned int *>(m_num) = 0; }
    CUDA_HOSTDEVICE unsigned int GetNum() const noexcept { return *reinterpret_cast<unsigned int *>(m_num); }

    CUDA_DEVICE unsigned int Push(const T &item) noexcept {
        unsigned int *num = reinterpret_cast<unsigned int *>(m_num);
        auto index = atomicAdd(num, 1);
        (*this)[index] = item;
        return index;
    }
#endif
};

#ifndef PUPIL_OPTIX
class DynamicArrayManager : public util::Singleton<DynamicArrayManager> {
private:
    std::unordered_map<CUdeviceptr, CUdeviceptr> m_cuda_dynamic_array_size;

public:
    template<typename T>
    [[nodiscard]] DynamicArray<T> GetDynamicArray(CUdeviceptr data, unsigned int num) noexcept {
        if (m_cuda_dynamic_array_size.find(data) == m_cuda_dynamic_array_size.end()) {
            m_cuda_dynamic_array_size[data] = cuda::CudaMemcpyToDevice(&num, sizeof(num));
        }

        DynamicArray<T> ret;
        ret.SetData(data, m_cuda_dynamic_array_size[data]);
        return ret;
    }

    template<typename T>
    void ClearDynamicArray(DynamicArray<T> &d_array) noexcept {
        unsigned int num = 0;
        cuda::CudaMemcpyToDevice(d_array.GetNum(), &num, sizeof(num));
    }

    template<typename T>
    [[nodiscard]] unsigned int GetDynamicArrayNum(DynamicArray<T> &d_array) noexcept {
        unsigned int ret;
        cuda::CudaMemcpyToHost(&ret, d_array.GetNum(), sizeof(ret));
        return ret;
    }

    void Clear() noexcept {
        for (auto &[data, size] : m_cuda_dynamic_array_size)
            CUDA_FREE(size);

        m_cuda_dynamic_array_size.clear();
    }
};
#endif
}// namespace Pupil::cuda