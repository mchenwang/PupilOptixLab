#pragma once

#include "cuda_util/vec_math.h"
#include "cuda_util/texture.h"

namespace optix_util {
enum class EEmitterType {
    None,
    TriArea,
    Sphere,
    // Point,
    EnvMap,
    ConstEnv
};

struct EmitterSampleRecord {
    float3 radiance;
    float3 wi;

    float distance;
    float pdf;
    bool is_delta;
};
struct EmitEvalRecord {
    float3 radiance;
    float pdf;
};
}// namespace optix_util

#include "area.h"
#include "sphere.h"
#include "env.h"