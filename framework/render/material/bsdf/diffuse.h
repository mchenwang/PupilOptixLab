#pragma once

#include "../predefine.h"
#include "cuda/texture.h"
#include "optix/util.h"

namespace Pupil::optix::material {

struct Diffuse {
    cuda::Texture reflectance;

    struct Local {
        float3 reflectance;
        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            float3 f = make_float3(0.f);
            if (record.wi.z > 0.f && record.wo.z > 0.f)
                f = reflectance * M_1_PIf;
            record.f = f;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            float pdf = 0.f;
            if (record.wi.z > 0.f && record.wo.z > 0.f)
                pdf = optix::CosineSampleHemispherePdf(record.wi);
            record.pdf = pdf;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float2 xi = record.sampler->Next2();
            record.wi = Pupil::optix::CosineSampleHemisphere(xi.x, xi.y);
            GetPdf(record);
            GetBsdf(record);
            record.sampled_type = EBsdfLobeType::DiffuseReflection;
        }
    };

    CUDA_DEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.reflectance = reflectance.Sample(sampled_tex);
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material