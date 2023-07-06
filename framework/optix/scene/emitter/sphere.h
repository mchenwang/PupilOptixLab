#pragma once

namespace Pupil::optix {
struct SphereEmitter {
    cuda::Texture radiance;

    float area;

    struct {
        float3 center;
        float radius;
    } geo;

    CUDA_DEVICE void SampleDirect(EmitterSampleRecord &ret, LocalGeometry &hit_geo, float2 xi) const noexcept {
        float3 t = Pupil::optix::UniformSampleSphere(xi.x, xi.y);
        float3 position = t * geo.radius + geo.center;
        float3 normal = normalize(t);
        float2 tex = Pupil::optix::GetSphereTexcoord(t);
        ret.radiance = radiance.Sample(tex);

        ret.wi = normalize(position - hit_geo.position);
        float NoL = dot(hit_geo.normal, ret.wi);
        float LNoL = dot(normal, -ret.wi);
        if (NoL > 0.f && LNoL > 0.f) {
            float distance = length(position - hit_geo.position);
            ret.pdf = distance * distance / (LNoL * area);
            ret.distance = distance;
        }

        ret.pos = position;
        ret.normal = normal;
    }

    CUDA_DEVICE void Eval(EmitEvalRecord &ret, LocalGeometry &emit_local_geo, float3 scatter_pos) const noexcept {
        float3 dir = normalize(scatter_pos - emit_local_geo.position);
        float LNoL = dot(emit_local_geo.normal, dir);
        if (LNoL > 0.f) {
            float distance = length(scatter_pos - emit_local_geo.position);
            ret.pdf = distance * distance / (LNoL * area);
            ret.radiance = radiance.Sample(emit_local_geo.texcoord);
        }
    }
};
}// namespace Pupil::optix