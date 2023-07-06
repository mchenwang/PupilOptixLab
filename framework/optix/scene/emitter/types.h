#pragma once

#include "cuda/vec_math.h"
#include "cuda/texture.h"

namespace Pupil::optix {
enum class EEmitterType : unsigned int {
    None,
    TriArea,
    Sphere,
    ConstEnv,
    EnvMap,
    // Point,
    // DistDir
};

struct EmitterSampleRecord {
    float3 radiance;
    float3 wi;
    float3 pos;
    float3 normal;

    float distance;
    float pdf = 0.f;
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