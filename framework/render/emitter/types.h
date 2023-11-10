#pragma once

#include "cuda/vec_math.h"
#include "texture.h"

namespace Pupil::optix {
    enum class EEmitterType : unsigned int {
        None,
        TriMesh,
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
        bool  is_delta;
    };
    struct EmitEvalRecord {
        unsigned int primitive_index;
        float3       radiance;
        float        pdf;
    };
}// namespace Pupil::optix

#include "area.h"
#include "sphere.h"
#include "env.h"