#include "material/optix_material.h"

#include "../fresnel.h"

using namespace Pupil::optix;
using namespace material;

#define PUPIL_MATERIAL_ATTR_DEFINE(attr)                                                                                                                      \
    extern "C" __device__ void PUPIL_MAT_SAMPLE_CALL(attr)(BsdfSampleRecord & ret, const material::Material &mat, float2 xi, float3 wo, float2 sampled_tex) { \
        ret = mat.attr.Sample(xi, wo, sampled_tex);                                                                                                           \
    }                                                                                                                                                         \
    extern "C" __device__ void PUPIL_MAT_EVAL_CALL(attr)(BsdfEvalRecord & ret, const material::Material &mat, float3 wi, float3 wo, float2 sampled_tex) {     \
        ret.f = mat.attr.GetBsdf(sampled_tex, wi, wo);                                                                                                        \
        ret.pdf = mat.attr.GetPdf(wi, wo);                                                                                                                    \
    }
#include "../material_decl.inl"
#undef PUPIL_MATERIAL_ATTR_DEFINE