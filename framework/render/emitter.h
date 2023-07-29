#pragma once

#include "cuda/preprocessor.h"
#include "cuda/texture.h"

#include "render/geometry.h"
#include "optix/util.h"
#include "emitter/types.h"

#include <optix.h>

namespace Pupil::optix {
struct Emitter {
    EEmitterType type CONST_STATIC_INIT(EEmitterType::None);
    float weight;
    float select_probability;

    union {
        TriAreaEmitter area;
        SphereEmitter sphere;
        ConstEnvEmitter const_env;
        EnvMapEmitter env_map;
    };

    CUDA_HOSTDEVICE Emitter() noexcept {}

#ifndef PUPIL_CPP
    CUDA_DEVICE float3 GetEnvCenter() const noexcept {
        switch (type) {
            case EEmitterType::ConstEnv:
                return const_env.center;
            case EEmitterType::EnvMap:
                return env_map.center;
        }
        return make_float3(0.f);
    }
    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
        switch (type) {
            case EEmitterType::TriArea:
                area.Eval(ret, emit_local_geo, scatter_pos);
                break;
            case EEmitterType::Sphere:
                sphere.Eval(ret, emit_local_geo, scatter_pos);
                break;
            case EEmitterType::ConstEnv:
                const_env.Eval(ret, emit_local_geo, scatter_pos);
                break;
            case EEmitterType::EnvMap:
                env_map.Eval(ret, emit_local_geo, scatter_pos);
                break;
        }
    }

    CUDA_DEVICE float3 GetRadiance(float2 tex) const noexcept {
        float3 ret;
        switch (type) {
            case EEmitterType::TriArea:
                ret = area.radiance.Sample(tex);
                break;
            case EEmitterType::Sphere:
                ret = sphere.radiance.Sample(tex);
                break;
            case EEmitterType::ConstEnv:
                ret = const_env.color;
                break;
            case EEmitterType::EnvMap:
                ret = env_map.radiance.Sample(tex);
                break;
        }
        return ret;
    }

    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
        switch (type) {
            case EEmitterType::TriArea:
                area.SampleDirect(ret, hit_geo, xi);
                break;
            case EEmitterType::Sphere:
                sphere.SampleDirect(ret, hit_geo, xi);
                break;
            case EEmitterType::ConstEnv:
                const_env.SampleDirect(ret, hit_geo, xi);
                break;
            case EEmitterType::EnvMap:
                env_map.SampleDirect(ret, hit_geo, xi);
                break;
        }
    }
#endif
#ifdef PUPIL_OPTIX
    CUDA_DEVICE static bool TraceShadowRay(OptixTraversableHandle ias,
                                           float3 ray_o, float3 ray_dir,
                                           float t_min, float t_max) noexcept {
        unsigned int occluded = 0u;
        optixTrace(ias, ray_o, ray_dir,
                   t_min, t_max, 0.f,
                   255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   1, 2, 1, occluded);
        return occluded;
    }
#endif
};

struct EmitterGroup {
    cuda::ConstArrayView<Emitter> areas;
    cuda::ConstArrayView<Emitter> points;
    cuda::ConstArrayView<Emitter> directionals;
    cuda::ConstDataView<Emitter> env;

    CUDA_HOSTDEVICE const Emitter &SelectOneEmiiter(float p) noexcept {
        unsigned int i = 0;
        float sum_p = 0.f;
        const Emitter *emitter_cb = nullptr;
        for (; i < areas.GetNum(); ++i) {
            if (p <= sum_p + areas[i].select_probability) {
                return areas[i];
            }
            sum_p += areas[i].select_probability;
            emitter_cb = &areas[i];
        }
        for (i = 0; i < points.GetNum(); ++i) {
            if (p <= sum_p + points[i].select_probability) {
                return points[i];
            }
            sum_p += points[i].select_probability;
            emitter_cb = &points[i];
        }
        for (i = 0; i < directionals.GetNum(); ++i) {
            if (p <= sum_p + directionals[i].select_probability) {
                return directionals[i];
            }
            sum_p += directionals[i].select_probability;
            emitter_cb = &directionals[i];
        }
        return env ? *env.operator->() : *emitter_cb;
    }
};

}// namespace Pupil::optix