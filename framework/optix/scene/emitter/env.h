#pragma once

namespace Pupil::optix {
struct EnvMapEmitter {
    cuda::Texture radiance;

    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
    }

    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
    }
};

struct ConstEnvEmitter {
    float3 color;
    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
        ret.wi = Pupil::optix::UniformSampleSphere(xi.x, xi.y);
        ret.pdf = 0.25f * M_1_PIf;
        ret.distance = Pupil::optix::MAX_DISTANCE;
        ret.radiance = color;
        ret.is_delta = false;

        // TODO
        // ret.pos = position;
        // ret.normal = normal;
    }

    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
        // float3 ray_dir = emit_local_geo.position - scatter_pos;
        // float2 tex = Pupil::optix::GetSphereTexcoord(make_float3(ray_dir.x, ray_dir.z, ray_dir.y));
        ret.radiance = color;
        ret.pdf = 0.25f * M_1_PIf;
    }
};
}// namespace Pupil::optix