#pragma once

#include "cuda/preprocessor.h"

namespace Pupil::optix {
enum class EBsdfLobeType : unsigned int {
    Unknown = 0,
    Null = 1,
    DiffuseReflection = 1 << 1,
    DiffuseTransmission = 1 << 2,
    GlossyReflection = 1 << 3,
    GlossyTransmission = 1 << 4,
    DeltaReflection = 1 << 5,
    DeltaTransmission = 1 << 6
};

struct BsdfSampleRecord {
    float3 f;
    float3 wi;
    float pdf;
    EBsdfLobeType lobe_type;
};
struct BsdfEvalRecord {
    float3 f;
    float pdf;
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
#include "conductor.h"
#include "rough_conductor.h"
#include "plastic.h"
#include "rough_plastic.h"