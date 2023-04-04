#pragma once

#include "cuda/preprocessor.h"
#include "cuda/texture.h"
#include "scene/emitter.h"

#include "optix/geometry.h"
#include "optix/util.h"
#include "optix/scene/emitter/types.h"

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
#include <vector>

namespace Pupil::scene {
class Scene;
}
#else
#include <optix.h>
#endif

namespace Pupil::optix {
struct Emitter {
    EEmitterType type CONST_STATIC_INIT(EEmitterType::None);
    float select_probability;

    union {
        TriAreaEmitter area;
        SphereEmitter sphere;
        ConstEnvEmitter const_env;
        EnvMapEmitter env_map;
    };

    CUDA_HOSTDEVICE Emitter() noexcept {}

    CUDA_HOSTDEVICE EmitEvalRecord Eval(LocalGeometry emit_local_geo, float3 scatter_pos) const noexcept {
        EmitEvalRecord ret;
        switch (type) {
            case EEmitterType::TriArea:
                ret = area.Eval(emit_local_geo, scatter_pos);
                break;
            case EEmitterType::Sphere:
                ret = sphere.Eval(emit_local_geo, scatter_pos);
                break;
            case EEmitterType::ConstEnv:
                ret = const_env.Eval(emit_local_geo, scatter_pos);
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
            case EEmitterType::ConstEnv:
                ret = const_env.color;
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
            case EEmitterType::ConstEnv:
                ret = const_env.SampleDirect(hit_geo, xi);
                break;
        }
        return ret;
    }

#ifndef PUPIL_OPTIX_LAUNCHER_SIDE
    CUDA_HOSTDEVICE static bool TraceShadowRay(OptixTraversableHandle ias,
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
        const Emitter &cb_emitter =
            env ? *env.operator->() :
                  (areas ? areas[0] :
                           (points ? points[0] : directionals[0]));
        for (; i < areas.GetNum(); ++i) {
            if (p > sum_p && p < sum_p + areas[i].select_probability) {
                return areas[i];
            }
            sum_p += areas[i].select_probability;
        }
        for (i = 0; i < points.GetNum(); ++i) {
            if (p > sum_p && p < sum_p + points[i].select_probability) {
                return points[i];
            }
            sum_p += points[i].select_probability;
        }
        for (i = 0; i < directionals.GetNum(); ++i) {
            if (p > sum_p && p < sum_p + directionals[i].select_probability) {
                return directionals[i];
            }
            sum_p += directionals[i].select_probability;
        }
        return cb_emitter;
    }
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE

class EmitterHelper {
private:
    std::vector<Emitter> m_areas;
    std::vector<Emitter> m_points;
    std::vector<Emitter> m_directionals;
    Emitter m_env;

    CUdeviceptr m_areas_cuda_memory;
    CUdeviceptr m_points_cuda_memory;
    CUdeviceptr m_directionals_cuda_memory;
    CUdeviceptr m_env_cuda_memory;

    void GenerateEmitters(scene::Scene *) noexcept;

public:
    EmitterHelper(scene::Scene *) noexcept;
    ~EmitterHelper() noexcept;

    void Clear() noexcept;
    void Reset(scene::Scene *) noexcept;
    EmitterGroup GetEmitterGroup() noexcept;
};

#endif
}// namespace Pupil::optix