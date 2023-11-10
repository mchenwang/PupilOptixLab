#pragma once

#include "texture.h"
#include "geometry.h"
#include "util.h"
#include "emitter/types.h"

#include <optix.h>

namespace Pupil::optix {
    struct Emitter {
        EEmitterType type CONST_STATIC_INIT(EEmitterType::None);
        float             weight;
        float             select_probability;

        union {
            TriMeshEmitter  tri_mesh;
            SphereEmitter   sphere;
            ConstEnvEmitter const_env;
            EnvMapEmitter   env_map;
        };

        CUDA_HOSTDEVICE Emitter() noexcept {}

#ifndef PUPIL_CPP
        CUDA_DEVICE void Eval(EmitEvalRecord& ret, LocalGeometry& emit_local_geo, float3 scatter_pos) const noexcept {
            switch (type) {
                case EEmitterType::TriMesh:
                    tri_mesh.Eval(ret, emit_local_geo, scatter_pos);
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
                case EEmitterType::TriMesh:
                    ret = tri_mesh.radiance.Sample(tex);
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

        CUDA_DEVICE void SampleDirect(EmitterSampleRecord& ret, LocalGeometry& hit_geo, float2 xi) const noexcept {
            switch (type) {
                case EEmitterType::TriMesh:
                    tri_mesh.SampleDirect(ret, hit_geo, xi);
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
                                               float3                 ray_o,
                                               float3                 ray_dir,
                                               float                  t_min,
                                               float                  t_max) noexcept {
            unsigned int occluded = 0u;
            optixTrace(ias, ray_o, ray_dir, t_min, t_max, 0.f, 255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);
            return occluded;
        }
#endif
    };

    /**
     * environment emitter is included in emitters
    */
    struct EmitterGroup {
        CUDA_HOSTDEVICE const Emitter* SelectOneEmiiter(float p) noexcept {
            unsigned int index = optix::Clamp<unsigned long long>(p * emitters.GetNum(), 0, emitters.GetNum() - 1);
            return emitters.GetNum() > 0 ? &emitters[index] : nullptr;
        }

        CUDA_HOSTDEVICE const Emitter* GetEnvironmentEmitter() noexcept {
            return env_index >= 0 ? &emitters[env_index] : nullptr;
        }

        CUDA_HOSTDEVICE const Emitter& operator[](unsigned int index) const noexcept {
            return emitters[index];
        }

        cuda::ConstArrayView<Emitter> emitters;
        int                           env_index;
    };

}// namespace Pupil::optix