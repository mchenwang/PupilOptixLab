#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/texture.h"
#include "scene/emitter.h"
#include "geometry.h"
#include "util.h"
#include "emitter/types.h"

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
#include <vector>

namespace scene {
class Scene;
}
#else
#include <optix.h>
#endif

namespace optix_util {
struct Emitter {
    EEmitterType type CONST_STATIC_INIT(EEmitterType::None);
    float select_probability;

    union {
        TriAreaEmitter area;
        SphereEmitter sphere;
    };

    CUDA_HOSTDEVICE Emitter() noexcept {}

    CUDA_HOSTDEVICE static const Emitter &SelectOneEmiiter(float p, const cuda::ConstArrayView<Emitter> &emitters) noexcept {
        unsigned int i = 0;
        float sum_p = 0.f;
        for (; i < emitters.GetNum() - 1; ++i) {
            if (p > sum_p && p < sum_p + emitters[i].select_probability) {
                break;
            }
            sum_p += emitters[i].select_probability;
        }
        return emitters[i];
    }

    CUDA_HOSTDEVICE EmitEvalRecord Eval(LocalGeometry emit_local_geo, float3 scatter_pos) const noexcept {
        EmitEvalRecord ret;
        switch (type) {
            case EEmitterType::TriArea:
                ret = area.Eval(emit_local_geo, scatter_pos);
                break;
            case EEmitterType::Sphere:
                ret = sphere.Eval(emit_local_geo, scatter_pos);
                break;
        }
        return ret;
    }

    CUDA_HOSTDEVICE float3 GetRadiance(float2 tex) const noexcept {
        float3 ret;
        switch (type) {
            case EEmitterType::TriArea:
                ret = area.radiance.Sample(tex);
                break;
            case EEmitterType::Sphere:
                ret = sphere.radiance.Sample(tex);
                break;
        }
        return ret;
    }

    CUDA_HOSTDEVICE EmitterSampleRecord SampleDirect(LocalGeometry hit_geo, float2 xi) const noexcept {
        EmitterSampleRecord ret;
        switch (type) {
            case EEmitterType::TriArea:
                ret = area.SampleDirect(hit_geo, xi);
                break;
            case EEmitterType::Sphere:
                ret = sphere.SampleDirect(hit_geo, xi);
                break;
        }
        return ret;
    }

    CUDA_HOSTDEVICE static bool TraceShadowRay(OptixTraversableHandle ias,
                                               float3 ray_o, float3 ray_dir,
                                               float t_min, float t_max) noexcept {
        unsigned int occluded = 0u;
#ifndef PUPIL_OPTIX_LAUNCHER_SIDE
        optixTrace(ias, ray_o, ray_dir,
                   t_min, t_max, 0.f,
                   255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   1, 2, 1, occluded);
#endif
        return occluded;
    }
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
std::vector<Emitter> GenerateEmitters(scene::Scene *) noexcept;
#endif
}// namespace optix_util