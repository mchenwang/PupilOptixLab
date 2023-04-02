#pragma once

namespace Pupil::optix {
struct EnvMapEmitter {
    cuda::Texture radiance;

    CUDA_HOSTDEVICE EmitterSampleRecord SampleDirect(LocalGeometry hit_geo, float2 xi) const noexcept {
        EmitterSampleRecord ret{};

        return ret;
    }

    CUDA_HOSTDEVICE EmitEvalRecord Eval(LocalGeometry emit_local_geo, float3 scatter_pos) const noexcept {
        EmitEvalRecord ret{};

        return ret;
    }
};

struct ConstEnvEmitter {
    float3 color;
    CUDA_HOSTDEVICE EmitterSampleRecord SampleDirect(LocalGeometry hit_geo, float2 xi) const noexcept {
        EmitterSampleRecord ret{};
        ret.wi = Pupil::optix::UniformSampleSphere(xi.x, xi.y);
        ret.pdf = 0.25f * M_1_PIf;
        ret.distance = Pupil::optix::MAX_DISTANCE;
        ret.radiance = color;
        ret.is_delta = false;

        return ret;
    }

    CUDA_HOSTDEVICE EmitEvalRecord Eval(LocalGeometry emit_local_geo, float3 scatter_pos) const noexcept {
        EmitEvalRecord ret{};
        // float3 ray_dir = emit_local_geo.position - scatter_pos;
        // float2 tex = Pupil::optix::GetSphereTexcoord(make_float3(ray_dir.x, ray_dir.z, ray_dir.y));
        ret.radiance = color;
        ret.pdf = 0.25f * M_1_PIf;

        return ret;
    }
};
}// namespace Pupil::optix