#pragma once

#include "cuda/data_view.h"

namespace Pupil::optix {
struct EnvMapEmitter {
    cuda::Texture radiance;
    float3 center;
    uint2 map_size;
    cuda::ConstArrayView<float> row_cdf;
    cuda::ConstArrayView<float> col_cdf;
    cuda::ConstArrayView<float> row_weight;

    struct {
        float3 r0;
        float3 r1;
        float3 r2;
    } to_world, to_local;

    float normalization;
    float scale;

    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
        unsigned int row_index = 0;
        for (; row_index < row_cdf.GetNum() - 1; ++row_index) {
            if (xi.x <= row_cdf[row_index]) break;
        }
        unsigned int col_index = 0;
        for (int i = row_index * (map_size.x + 1); col_index < map_size.x - 1; ++i, ++col_index) {
            if (xi.y <= col_cdf[i]) break;
        }

        const float phi = col_index * M_PIf * 2.f / map_size.x;
        const float theta = row_index * M_PIf / map_size.y;

        const auto local_wi = make_float3(sin(theta) * sinf(M_PIf - phi), cos(theta), sin(theta) * cosf(M_PIf - phi));

        ret.wi = make_float3(dot(to_world.r0, local_wi), dot(to_world.r1, local_wi), dot(to_world.r2, local_wi));
        ret.distance = Pupil::optix::MAX_DISTANCE;
        const float2 tex = make_float2(phi * 0.5f * M_1_PIf, theta * M_1_PIf);
        ret.radiance = radiance.Sample(tex) * scale;
        ret.is_delta = false;

        ret.pdf = Pupil::optix::GetLuminance(ret.radiance) * row_weight[row_index] * normalization /
                  max(1e-4f, abs(sin(theta)));
        if (ret.pdf < 0.f) ret.pdf = 0.f;
        ret.pos = hit_geo.position + ret.wi * ret.distance;
        ret.normal = normalize(center - ret.pos);
    }

    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
        float3 dir = normalize(emit_local_geo.position - scatter_pos);
        dir = make_float3(dot(to_local.r0, dir), dot(to_local.r1, dir), dot(to_local.r2, dir));
        const float phi = M_PIf - atan2(dir.x, dir.z);
        const float theta = acos(dir.y);
        const float2 tex = make_float2(phi * 0.5f * M_1_PIf, theta * M_1_PIf);

        unsigned int row_index = static_cast<unsigned int>(tex.y * map_size.y);
        row_index = clamp(row_index, 0u, map_size.y - 2u);
        ret.radiance = radiance.Sample(tex) * scale;
        ret.pdf = optix::GetLuminance(ret.radiance) *
                  optix::Lerp(row_weight[row_index], row_weight[row_index + 1], tex.y * map_size.y - 1.f * row_index) *
                  normalization / max(1e-4f, abs(sin(theta)));
    }
};

struct ConstEnvEmitter {
    float3 color;
    float3 center;
    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
        float3 local_wi = Pupil::optix::UniformSampleHemisphere(xi.x, xi.y);
        ret.wi = Pupil::optix::ToWorld(local_wi, hit_geo.normal);
        ret.pdf = Pupil::optix::UniformSampleHemispherePdf(local_wi);
        ret.distance = Pupil::optix::MAX_DISTANCE;
        ret.radiance = color;
        ret.is_delta = false;

        ret.pos = hit_geo.position + ret.wi * ret.distance;
        ret.normal = normalize(center - ret.pos);
    }

    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
        ret.pdf = 0.25f * M_1_PIf;
        ret.radiance = color;
    }
};
}// namespace Pupil::optix