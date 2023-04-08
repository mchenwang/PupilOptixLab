#pragma once

#include "preprocessor.h"
#include <cuda.h>

namespace Pupil::cuda {
// cuda memory needs to be released explicitly
template<typename T>
class ConstArrayView {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);
    size_t m_num CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE ConstArrayView() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data, size_t num) noexcept {
        m_data = cuda_data;
        m_num = num;
    }
    CUDA_HOSTDEVICE const T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE size_t GetNum() const noexcept { return m_num; }
    CUDA_HOSTDEVICE const T &operator[](unsigned int index) const noexcept {
        return *reinterpret_cast<T *>(m_data + index * sizeof(T));
    }
};

// cuda memory needs to be released explicitly
template<typename T>
class ConstDataView {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE ConstDataView() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data) noexcept { m_data = cuda_data; }
    CUDA_HOSTDEVICE const T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE const T *operator->() const noexcept {
        return reinterpret_cast<T *>(m_data);
    }
};

template<typename T>
class RWArrayView {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);
    size_t m_num CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE RWArrayView() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data, size_t num) noexcept {
        m_data = cuda_data;
        m_num = num;
    }
    CUDA_HOSTDEVICE T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE size_t GetNum() const noexcept { return m_num; }
    CUDA_HOSTDEVICE T &operator[](unsigned int index) const noexcept {
        return *reinterpret_cast<T *>(m_data + index * sizeof(T));
    }
};

template<typename T>
class RWDataView {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE RWDataView() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data) noexcept { m_data = cuda_data; }
    CUDA_HOSTDEVICE T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE T *operator->() const noexcept {
        return reinterpret_cast<T *>(m_data);
    }
};
}// namespace Pupil::cuda