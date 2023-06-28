#pragma once

#include "preprocessor.h"
#include "vec_math.h"

namespace Pupil::cuda {
/// @brief generate random float [0, 1)
class Random {
private:
    unsigned int m_seed;

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
    CUDA_HOSTDEVICE void SetSeed(unsigned int seed) noexcept { m_seed = seed; }

    CUDA_HOSTDEVICE float Next() noexcept {
        const unsigned int LCG_A = 1664525u;
        const unsigned int LCG_C = 1013904223u;
        m_seed = (LCG_A * m_seed + LCG_C);
        return static_cast<float>(m_seed & 0x00FFFFFF) / 0x01000000;
    }

    CUDA_HOSTDEVICE float2 Next2() noexcept {
        return make_float2(Next(), Next());
    }
};
}// namespace Pupil::cuda