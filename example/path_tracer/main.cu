#include <optix.h>
#include "type.h"

#include "render/util.h"
#include "render/geometry.h"
#include "render/material/bsdf/bsdf.h"

#include <cuda_runtime.h>

using namespace Pupil;

extern "C" {
__constant__ pt::OptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix::LocalGeometry       geo;
    optix::Material::LocalBsdf bsdf;
};

struct PathPayloadRecord {
    float3       radiance;
    cuda::Random random;

    float3 throughput;

    float3 ray_dir;
    float3 ray_o;

    optix::EBsdfLobeType bsdf_sample_type;
    float                bsdf_sample_pdf;

    HitInfo hit;

    unsigned int depth;
    unsigned int pixel_index;

    bool done;
};

struct NEERecord {
    unsigned int shadow_ray;
    float        shadow_ray_t_max;
    float3       shadow_ray_dir;
    float3       shadow_ray_o;
    float3       radiance;
};

__forceinline__ __device__ void ScatterRays(const unsigned int pixel_index, PathPayloadRecord* record, NEERecord* nee) noexcept {
    float rr = record->depth > 2 ? 0.95 : 1.0;
    if (record->random.Next() > rr) {
        record->done = true;
        return;
    }

    auto& geo  = record->hit.geo;
    auto& bsdf = record->hit.bsdf;

    // direct lighting
    {
        auto emitter = optix_launch_params.emitters.SelectOneEmiiter(record->random.Next());

        Pupil::optix::EmitterSampleRecord emitter_sample_record;
        emitter->SampleDirect(emitter_sample_record, geo, record->random.Next2());

        optix::BsdfSamplingRecord eval_record;
        eval_record.wi      = optix::ToLocal(emitter_sample_record.wi, geo.normal);
        eval_record.wo      = optix::ToLocal(-record->ray_dir, geo.normal);
        eval_record.sampler = &record->random;
        bsdf.Eval(eval_record);

        float3 bsdf_eval_f   = eval_record.f;
        float  bsdf_eval_pdf = eval_record.pdf;

        float emit_pdf  = emitter_sample_record.pdf * emitter->select_probability;
        nee->shadow_ray = 0;
        if (optix::IsValid(emit_pdf)) {
            nee->shadow_ray       = 1;
            nee->shadow_ray_dir   = emitter_sample_record.wi;
            nee->shadow_ray_o     = geo.position;
            nee->shadow_ray_t_max = emitter_sample_record.distance - 0.0001f;

            float NoL = abs(dot(geo.normal, emitter_sample_record.wi));
            float mis = emitter_sample_record.is_delta ? 1.f : optix::MISWeight(emitter_sample_record.pdf, bsdf_eval_pdf);

            nee->radiance = emitter_sample_record.radiance * record->throughput * bsdf_eval_f * NoL * mis / emit_pdf;
        }
    }

    // bsdf sampling
    {
        optix::BsdfSamplingRecord bsdf_sample_record;
        bsdf_sample_record.wo      = optix::ToLocal(-record->ray_dir, geo.normal);
        bsdf_sample_record.sampler = &record->random;
        bsdf.Sample(bsdf_sample_record);

        if (optix::IsValid(bsdf_sample_record.pdf)) {
            record->bsdf_sample_pdf  = bsdf_sample_record.pdf;
            record->bsdf_sample_type = bsdf_sample_record.sampled_type;
            record->throughput *= bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf;

            record->ray_dir = optix::ToWorld(bsdf_sample_record.wi, geo.normal);
            record->ray_o   = geo.position;
        } else {
            record->done = true;
        }
    }
}

extern "C" __global__ void __raygen__main() {
    const uint3        index       = optixGetLaunchIndex();
    const unsigned int w           = optix_launch_params.config.frame.width;
    const unsigned int h           = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    const auto& camera = optix_launch_params.camera;

    PathPayloadRecord record{};
    uint32_t          u0, u1;
    optix::PackPointer(&record, u0, u1);

    record.done        = false;
    record.depth       = 0u;
    record.throughput  = make_float3(1.f);
    record.radiance    = make_float3(0.f);
    record.pixel_index = pixel_index;
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());

    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = camera.sample_to_camera * point_on_film;

    d /= d.w;
    d.w = 0.f;
    d   = normalize(d);

    record.ray_dir = normalize(make_float3(camera.camera_to_world * d));

    record.ray_o = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    NEERecord nee;
    for (; record.depth < optix_launch_params.config.max_depth && !record.done; ++record.depth) {
        optixTrace(optix_launch_params.handle,
                   record.ray_o, record.ray_dir,
                   0.0001f, 1e16f, 0.f,
                   255, OPTIX_RAY_FLAG_NONE,
                   0, 2, 0,
                   u0, u1);
        if (record.depth == optix_launch_params.config.max_depth - 1 || record.done)
            break;

        ScatterRays(pixel_index, &record, &nee);

        unsigned int occluded = 0u;
        optixTrace(optix_launch_params.handle,
                   nee.shadow_ray_o, nee.shadow_ray_dir,
                   0.0001f, nee.shadow_ray_t_max, 0.f,
                   255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   1, 2, 1,
                   occluded);
        if (nee.shadow_ray && !record.done) {
            if (!occluded) record.radiance += nee.radiance;
        }
    }

    if (optix_launch_params.config.accumulated_flag && optix_launch_params.sample_cnt > 0) {
        const float  t   = 1.f / (optix_launch_params.sample_cnt + 1.f);
        const float3 pre = make_float3(optix_launch_params.accum_buffer[pixel_index]);
        record.radiance  = lerp(pre, record.radiance, t);
    }
    optix_launch_params.accum_buffer[pixel_index] = make_float4(record.radiance, 1.f);
    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (auto env = optix_launch_params.emitters.GetEnvironmentEmitter();
        env) {
        const auto ray_dir = record->ray_dir;
        const auto ray_o   = record->ray_o;

        optix::LocalGeometry env_local;
        env_local.position = ray_o + ray_dir;
        optix::EmitEvalRecord emit_record;
        env->Eval(emit_record, env_local, ray_o);

        if (record->depth > 0) {
            float mis = optix::MISWeight(record->bsdf_sample_pdf, emit_record.pdf);
            record->radiance += record->throughput * emit_record.radiance * mis;
        } else {
            record->radiance = emit_record.radiance;

            optix_launch_params.normal_buffer[record->pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            optix_launch_params.albedo_buffer[record->pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
        }
    }
    record->done = true;
}

__device__ __forceinline__ void ClosestHit() {
    const auto* sbt_data = reinterpret_cast<pt::HitGroupData*>(optixGetSbtDataPointer());
    auto        record   = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = record->ray_dir;
    const auto ray_o   = record->ray_o;

    sbt_data->geo.GetHitLocalGeometry(record->hit.geo, ray_dir, sbt_data->mat.twosided);
    record->hit.bsdf = sbt_data->mat.GetLocalBsdf(record->hit.geo.texcoord);

    // record->radiance = record->hit.geo.normal;

    if (record->depth == 0) {
        if (sbt_data->emitter_index >= 0) {
            auto& emitter = optix_launch_params.emitters[sbt_data->emitter_index];
            record->radiance += emitter.GetRadiance(record->hit.geo.texcoord);
        }

        optix_launch_params.normal_buffer[record->pixel_index] = make_float4(record->hit.geo.normal, 1.f);
        optix_launch_params.albedo_buffer[record->pixel_index] = make_float4(record->hit.bsdf.GetAlbedo(), 1.f);

        return;
    }

    if (sbt_data->emitter_index < 0) return;
    auto& emitter = optix_launch_params.emitters[sbt_data->emitter_index];

    optix::EmitEvalRecord emit_record;
    emit_record.primitive_index = optixGetPrimitiveIndex();
    emitter.Eval(emit_record, record->hit.geo, ray_o);

    float mis = record->bsdf_sample_type & optix::EBsdfLobeType::Delta ?
                    1.f :
                    optix::MISWeight(record->bsdf_sample_pdf, emit_record.pdf * emitter.select_probability);

    record->radiance += emit_record.radiance * mis * record->throughput;
}

extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}

__device__ __forceinline__ void ClosestHitShadow() {
    optixSetPayload_0(1u);
}

extern "C" __global__ void __closesthit__default() { ClosestHit(); }
extern "C" __global__ void __closesthit__default_sphere() { ClosestHit(); }
extern "C" __global__ void __closesthit__default_curve() { ClosestHit(); }

extern "C" __global__ void __closesthit__shadow() { ClosestHitShadow(); }
extern "C" __global__ void __closesthit__shadow_sphere() { ClosestHitShadow(); }
extern "C" __global__ void __closesthit__shadow_curve() { ClosestHitShadow(); }