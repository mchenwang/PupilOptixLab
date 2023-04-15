#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ pt::OptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material mat;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 radiance;
    float3 env_radiance;
    float env_pdf;
    cuda::Random random;

    float3 throughput;

    HitInfo hit;

    unsigned int depth;
    bool done;
};

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    auto &camera = *optix_launch_params.camera.GetDataPtr();

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);

    record.done = false;
    record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.env_radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());

    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
    // const float2 subpixel = make_float2((static_cast<float>(index.x)) / w, (static_cast<float>(index.y)) / h);
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = camera.sample_to_camera * point_on_film;

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

    float3 ray_origin = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    int depth = 0;
    auto local_hit = record.hit;

    while (!record.done) {
        if (depth == 0) {
            if (record.hit.emitter_index >= 0) {
                auto &emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
                auto emission = emitter.GetRadiance(local_hit.geo.texcoord);
                record.radiance += emission;
            }
        }

        ++depth;
        if (depth >= optix_launch_params.config.max_depth)
            break;

        // direct light sampling
        {
            auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(record.random.Next());
            auto emitter_sample_record = emitter.SampleDirect(local_hit.geo, record.random.Next2());

            if (!optix::IsZero(emitter_sample_record.pdf)) {
                bool occluded =
                    optix::Emitter::TraceShadowRay(
                        optix_launch_params.handle,
                        local_hit.geo.position, emitter_sample_record.wi,
                        0.001f, emitter_sample_record.distance - 0.001f);
                if (!occluded) {
                    float3 wi = optix::ToLocal(emitter_sample_record.wi, local_hit.geo.normal);
                    float3 wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
                    auto [f, pdf] = record.hit.mat.Eval(wi, wo, local_hit.geo.texcoord);
                    if (!optix::IsZero(f)) {
                        float NoL = dot(local_hit.geo.normal, emitter_sample_record.wi);
                        float mis = emitter_sample_record.is_delta ? 1.f : optix::MISWeight(emitter_sample_record.pdf, pdf);
                        emitter_sample_record.pdf *= emitter.select_probability;
                        record.radiance += record.throughput * emitter_sample_record.radiance * f * NoL * mis / emitter_sample_record.pdf;
                    }
                }
            }
        }
        // bsdf sampling
        {
            float3 wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
            auto bsdf_sample_record = record.hit.mat.Sample(record.random.Next2(), wo, local_hit.geo.texcoord);
            if (optix::IsZero(bsdf_sample_record.f * abs(bsdf_sample_record.wi.z)) || optix::IsZero(bsdf_sample_record.pdf))
                break;

            record.throughput *= bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf;

            float rr = depth > 2 ? 0.95 : 1.0;
            if (record.random.Next() > rr)
                break;
            record.throughput /= rr;

            ray_origin = record.hit.geo.position;
            ray_direction = optix::ToWorld(bsdf_sample_record.wi, local_hit.geo.normal);

            optixTrace(optix_launch_params.handle,
                       ray_origin, ray_direction,
                       0.001f, 1e16f, 0.f,
                       255, OPTIX_RAY_FLAG_NONE,
                       0, 2, 0,
                       u0, u1);

            if (record.done) {
                float mis = optix::MISWeight(bsdf_sample_record.pdf, record.env_pdf);
                record.env_radiance *= record.throughput * mis;
                break;
            }

            local_hit = record.hit;
            if (record.hit.emitter_index >= 0) {
                auto &emitter = optix_launch_params.emitters.areas[record.hit.emitter_index];
                auto emit_record = emitter.Eval(record.hit.geo, ray_origin);

                if (!optix::IsZero(emit_record.pdf)) {
                    float mis = bsdf_sample_record.lobe_type & optix::EBsdfLobeType::DeltaReflection ?
                                    1.f :
                                    optix::MISWeight(bsdf_sample_record.pdf, emit_record.pdf * emitter.select_probability);

                    record.radiance += record.throughput * emit_record.radiance * mis;
                }
            }
        }
    }
    record.radiance += record.env_radiance;

    if (optix_launch_params.sample_cnt > 0) {
        const float t = 1.f / (optix_launch_params.sample_cnt + 1.f);
        const float3 pre = make_float3(optix_launch_params.accum_buffer[pixel_index]);
        record.radiance = lerp(pre, record.radiance, t);
    }
    optix_launch_params.accum_buffer[pixel_index] = make_float4(record.radiance, 1.f);
    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
        optix::LocalGeometry temp;
        temp.position = optixGetWorldRayDirection();
        float3 scatter_pos = make_float3(0.f);
        auto env_emit_record = optix_launch_params.emitters.env->Eval(temp, scatter_pos);
        record->env_radiance = env_emit_record.radiance;
        record->env_pdf = env_emit_record.pdf;
    }
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
extern "C" __global__ void __closesthit__default() {
    const pt::HitGroupData *sbt_data = (pt::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    record->hit.geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    } else {
        record->hit.emitter_index = -1;
    }

    record->hit.mat = sbt_data->mat;
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}