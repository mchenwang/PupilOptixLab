#pragma once

namespace optix_util {
struct BsdfSampleRecord {
    float3 f;
    float3 wi;
    float pdf;
};
struct BsdfEvalRecord {
    float3 f;
    float pdf;
};
}// namespace optix_util

#include "diffuse.h"
#include "dielectric.h"
#include "conductor.h"
#include "rough_conductor.h"
#include "plastic.h"
#include "rough_plastic.h"