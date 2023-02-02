#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/texture.h"
#include "scene/emitter.h"
#include <vector_types.h>

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
#include <vector>

namespace scene {
class Scene;
}
#endif

namespace optix_util {
enum class EEmitterType {
    None,
    Triangle,
    Sphere
};

struct TriangleEmitter {
    cuda::Texture radiance;

    float area;
    float select_probability;

    struct {
        struct {
            float3 pos;
            float3 normal;
            float2 tex;
        } v0, v1, v2;
    } geo;
};

struct SphereEmitter {
    cuda::Texture radiance;

    float area;
    float select_probability;

    struct {
        float3 center;
        float radius;
    } geo;
};

struct Emitter {
    EEmitterType type CONST_STATIC_INIT(EEmitterType::None);
    union {
        TriangleEmitter triangle;
        SphereEmitter sphere;
    };

    CUDA_HOSTDEVICE Emitter() noexcept {}
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
std::vector<Emitter> GenerateEmitters(const scene::Scene *) noexcept;
#endif
}// namespace optix_util