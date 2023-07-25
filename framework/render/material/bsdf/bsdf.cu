#include "../optix_material.h"
#include "../fresnel.h"

using namespace Pupil::optix;
using namespace material;

#define PUPIL_MATERIAL_ATTR_DEFINE(attr)                                                                                             \
    extern "C" __device__ void PUPIL_MAT_SAMPLE_CALL(attr)(BsdfSamplingRecord & record, const material::Material::LocalBsdf &bsdf) { \
        bsdf.attr.Sample(record);                                                                                                    \
    }                                                                                                                                \
    extern "C" __device__ void PUPIL_MAT_EVAL_CALL(attr)(BsdfSamplingRecord & record, const material::Material::LocalBsdf &bsdf) {   \
        bsdf.attr.GetBsdf(record);                                                                                                   \
        bsdf.attr.GetPdf(record);                                                                                                    \
    }
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_ATTR_DEFINE