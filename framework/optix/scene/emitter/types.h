#pragma once

#include "cuda/vec_math.h"
#include "cuda/texture.h"

namespace Pupil::optix {
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
}// namespace Pupil::optix

#include "area.h"
#include "sphere.h"
#include "env.h"