#pragma once

#ifndef PUPIL_OPTIX
#include "util/math.h"
#endif

#include "vec_math.h"
#include <cuda.h>

#include <vector_functions.h>

struct mat4x4 {
    float4 r0, r1, r2, r3;
};

CUDA_INLINE CUDA_HOSTDEVICE float4 operator*(const mat4x4& m, const float4& v) noexcept {
    return make_float4(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}

namespace Pupil::cuda {
#ifndef PUPIL_OPTIX
    inline float2 MakeFloat2(float x, float y) noexcept { return ::make_float2(x, y); }
    inline float3 MakeFloat3(float x, float y, float z) noexcept { return ::make_float3(x, y, z); }
    inline float4 MakeFloat4(float x, float y, float z, float w) noexcept { return ::make_float4(x, y, z, w); }
    inline float2 MakeFloat2(const Float2& v) noexcept { return ::make_float2(v.x, v.y); }
    inline float2 MakeFloat2(const Float3& v) noexcept { return ::make_float2(v.x, v.y); }
    inline float2 MakeFloat2(const Float4& v) noexcept { return ::make_float2(v.x, v.y); }
    inline float3 MakeFloat3(const Float3& v) noexcept { return ::make_float3(v.x, v.y, v.z); }
    inline float3 MakeFloat3(const Float4& v) noexcept { return ::make_float3(v.x, v.y, v.z); }
    inline float4 MakeFloat4(const Float4& v) noexcept { return ::make_float4(v.x, v.y, v.z, v.w); }
    inline mat4x4 MakeMat4x4(const Matrix4x4f& m) noexcept {
        mat4x4 ret;
        ret.r0 = MakeFloat4(m.r0);
        ret.r1 = MakeFloat4(m.r1);
        ret.r2 = MakeFloat4(m.r2);
        ret.r3 = MakeFloat4(m.r3);
        return ret;
    }
#endif

    /// @brief generate random float [0, 1)
    class Random {
    public:
        CUDA_HOSTDEVICE Random() noexcept : m_seed(0) {}

        CUDA_HOSTDEVICE void Init(unsigned int N, unsigned int val0, unsigned int val1) noexcept {
            unsigned int v0 = val0;
            unsigned int v1 = val1;
            unsigned int s0 = 0;

            for (unsigned int n = 0; n < N; n++) {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }

            m_seed = v0;
        }

        CUDA_HOSTDEVICE unsigned int GetSeed() const noexcept { return m_seed; }
        CUDA_HOSTDEVICE void         SetSeed(unsigned int seed) noexcept { m_seed = seed; }

        CUDA_HOSTDEVICE float Next() noexcept {
            const unsigned int LCG_A = 1664525u;
            const unsigned int LCG_C = 1013904223u;
            m_seed                   = (LCG_A * m_seed + LCG_C);
            return static_cast<float>(m_seed & 0x00FFFFFF) / 0x01000000;
        }

        CUDA_HOSTDEVICE float2 Next2() noexcept {
            return make_float2(Next(), Next());
        }

    private:
        unsigned int m_seed;
    };

    // cuda memory needs to be released explicitly
    template<typename T>
    class ConstArrayView {
    private:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
        size_t m_num       CONST_STATIC_INIT(0);

    public:
        CUDA_HOSTDEVICE ConstArrayView() noexcept {}

        CUDA_HOST void SetData(CUdeviceptr cuda_data, size_t num) noexcept {
            m_data = cuda_data;
            m_num  = num;
        }
        CUDA_HOSTDEVICE const T* GetDataPtr() const noexcept { return reinterpret_cast<T*>(m_data); }

        CUDA_HOSTDEVICE          operator bool() const noexcept { return m_data != 0; }
        CUDA_HOSTDEVICE size_t   GetNum() const noexcept { return m_num; }
        CUDA_HOSTDEVICE const T& operator[](unsigned int index) const noexcept {
            return *reinterpret_cast<T*>(m_data + index * sizeof(T));
        }
    };

    // cuda memory needs to be released explicitly
    template<typename T>
    class ConstDataView {
    public:
        CUDA_HOSTDEVICE ConstDataView() noexcept {}

        CUDA_HOST void           SetData(CUdeviceptr cuda_data) noexcept { m_data = cuda_data; }
        CUDA_HOSTDEVICE const T* GetDataPtr() const noexcept { return reinterpret_cast<T*>(m_data); }

        CUDA_HOSTDEVICE          operator bool() const noexcept { return m_data != 0; }
        CUDA_HOSTDEVICE const T* operator->() const noexcept {
            return reinterpret_cast<T*>(m_data);
        }

    private:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
    };

    template<typename T>
    class RWArrayView {
    public:
        CUDA_HOSTDEVICE RWArrayView() noexcept {}

        CUDA_HOST void SetData(CUdeviceptr cuda_data, size_t num) noexcept {
            m_data = cuda_data;
            m_num  = num;
        }
        CUDA_HOSTDEVICE T* GetDataPtr() const noexcept { return reinterpret_cast<T*>(m_data); }

        CUDA_HOSTDEVICE        operator bool() const noexcept { return m_data != 0; }
        CUDA_HOSTDEVICE size_t GetNum() const noexcept { return m_num; }
        CUDA_HOSTDEVICE T&     operator[](unsigned int index) const noexcept {
            return *reinterpret_cast<T*>(m_data + index * sizeof(T));
        }

    private:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
        size_t m_num       CONST_STATIC_INIT(0);
    };

    template<typename T>
    class RWDataView {
    public:
        CUDA_HOSTDEVICE RWDataView() noexcept {}

        CUDA_HOST void     SetData(CUdeviceptr cuda_data) noexcept { m_data = cuda_data; }
        CUDA_HOSTDEVICE T* GetDataPtr() const noexcept { return reinterpret_cast<T*>(m_data); }

        CUDA_HOSTDEVICE    operator bool() const noexcept { return m_data != 0; }
        CUDA_HOSTDEVICE T* operator->() const noexcept {
            return reinterpret_cast<T*>(m_data);
        }

    private:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
    };

    template<typename T>
    class DynamicArray {
    public:
        CUDA_HOSTDEVICE DynamicArray() noexcept {}

        CUDA_HOST void SetData(CUdeviceptr cuda_data, CUdeviceptr num) noexcept {
            m_data = cuda_data;
            m_num  = num;
        }
        CUDA_HOSTDEVICE T* GetDataPtr() const noexcept { return reinterpret_cast<T*>(m_data); }

        CUDA_HOSTDEVICE    operator bool() const noexcept { return m_data != 0; }
        CUDA_HOSTDEVICE T& operator[](unsigned int index) const noexcept {
            return *reinterpret_cast<T*>(m_data + index * sizeof(T));
        }

#ifdef PUPIL_CPP
        CUDA_HOSTDEVICE CUdeviceptr GetNum() const noexcept { return m_num; }
#else
        CUDA_DEVICE void             Clear() noexcept { *reinterpret_cast<unsigned int*>(m_num) = 0; }
        CUDA_HOSTDEVICE unsigned int GetNum() const noexcept { return *reinterpret_cast<unsigned int*>(m_num); }

        CUDA_DEVICE unsigned int Push(const T& item) noexcept {
            unsigned int* num   = reinterpret_cast<unsigned int*>(m_num);
            auto          index = atomicAdd(num, 1);
            (*this)[index]      = item;
            return index;
        }
#endif
    private:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
        CUdeviceptr m_num  CONST_STATIC_INIT(0);
    };
}// namespace Pupil::cuda