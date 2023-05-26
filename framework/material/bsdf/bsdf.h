#pragma once

#include "cuda/preprocessor.h"
#include "cuda/random.h"

namespace Pupil::optix {
enum class EBsdfLobeType : unsigned int {
    Unknown = 0,
    Null = 1,
    DiffuseReflection = 1 << 1,
    DiffuseTransmission = 1 << 2,
    GlossyReflection = 1 << 3,
    GlossyTransmission = 1 << 4,
    DeltaReflection = 1 << 5,
    DeltaTransmission = 1 << 6,

    Reflection = DiffuseReflection | GlossyReflection | DeltaReflection,
    Transmission = DiffuseTransmission | GlossyTransmission | DeltaTransmission,
    Diffuse = DiffuseReflection | DiffuseTransmission,
    Glossy = GlossyReflection | GlossyTransmission,
    Delta = DeltaReflection | DeltaTransmission,

    All = Reflection | Transmission | Null
};

struct BsdfSamplingRecord {
    float2 sampled_tex;
    float3 wi;
    float3 wo;
    float3 f = make_float3(0.f);
    float pdf = 0.f;

    cuda::Random *sampler = nullptr;

    EBsdfLobeType sampled_type = EBsdfLobeType::Unknown;
};

CUDA_INLINE CUDA_HOSTDEVICE bool BsdfLobeTypeMatch(EBsdfLobeType target, EBsdfLobeType type) {
    return static_cast<unsigned int>(target) & static_cast<unsigned int>(type);
}
CUDA_INLINE CUDA_HOSTDEVICE bool operator&(EBsdfLobeType target, EBsdfLobeType type) {
    return static_cast<unsigned int>(target) & static_cast<unsigned int>(type);
}
}// namespace Pupil::optix

#include "diffuse.h"
#include "dielectric.h"
#include "rough_dielectric.h"
#include "conductor.h"
#include "rough_conductor.h"
#include "plastic.h"
#include "rough_plastic.h"