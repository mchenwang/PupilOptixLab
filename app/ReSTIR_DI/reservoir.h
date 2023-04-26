#pragma once

#include <vector_types.h>
#include "cuda/preprocessor.h"
#include "cuda/random.h"

struct Reservoir {
    struct Sample {
        float3 pos;
        float3 normal;
        float3 emission;
        float distance;
        float3 radiance;
        float p_hat;
    };

    Sample y;
    float w_sum;
    float W;
    unsigned int M;

    CUDA_HOSTDEVICE Reservoir() noexcept { Init(); }

    CUDA_HOSTDEVICE void Init() noexcept {
        w_sum = 0.f;
        W = 0.f;
        M = 0u;
        y.p_hat = 0.f;
        y.radiance = make_float3(0.f);
    }

    CUDA_HOSTDEVICE void Update(
        const Sample &x_i, float w_i,
        Pupil::cuda::Random &random) noexcept {
        w_sum += w_i;
        M += 1;
        // if (random.Next() < w_i / w_sum)
        //     y = x_i;
        if (M == 1) {
            if (w_i != 0.f)// 第一个样本一定选
                y = x_i;
            else// 第一个样本选择的概率为0，去掉该样本
                M = 0;
        } else if (random.Next() < w_i / max(0.0001f, w_sum))
            y = x_i;
    }

    CUDA_HOSTDEVICE void CalcW() noexcept {
        W = (y.p_hat == 0.f || M == 0) ? 0.f : w_sum / (y.p_hat * M);
    }

    CUDA_HOSTDEVICE void Combine(const Reservoir &other, Pupil::cuda::Random &random) noexcept {
        Update(other.y, other.y.p_hat * other.W * other.M, random);
        M += other.M - 1;
        CalcW();
    }
};