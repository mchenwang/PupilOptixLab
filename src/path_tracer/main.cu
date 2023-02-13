#include <optix.h>
#include "type.h"

#include "optix_util/util.h"
#include "optix_util/geometry.h"

#include "cuda_util/random.h"

extern "C" {
__constant__ OptixLaunchParams optix_launch_params;
}

struct PathPayloadRecord {
    float3 radiance;
    cuda::Random random;

    float3 throughput;

    float3 hit_p;
    float3 wi;
    float bsdf_sampled_pdf;

    unsigned int depth;
    bool done;
};

extern "C" __global__ void __raygen__main() {
    // const RayGenData *sbt_data = (RayGenData *)optixGetSbtDataPointer();
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    auto &camera = *optix_launch_params.camera.operator->();

    // optix_launch_params.frame_buffer[pixel_index] =
    //     make_float4(
    //         (float)index.x / w,
    //         (float)index.y / h, 0.f, 1.f);

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix_util::PackPointer(&record, u0, u1);

    record.done = false;
    record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.bsdf_sampled_pdf = 0.f;
    record.random.Init(4, pixel_index, optix_launch_params.frame_cnt);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());

    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
    // const float2 subpixel = make_float2((static_cast<float>(index.x)) / w, (static_cast<float>(index.y)) / h);
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = make_float4(
        dot(camera.sample_to_camera.r0, point_on_film),
        dot(camera.sample_to_camera.r1, point_on_film),
        dot(camera.sample_to_camera.r2, point_on_film),
        dot(camera.sample_to_camera.r3, point_on_film));

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = (make_float3(
        dot(camera.camera_to_world.r0, d),
        dot(camera.camera_to_world.r1, d),
        dot(camera.camera_to_world.r2, d)));

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

    while (record.depth < optix_launch_params.config.max_depth - 1) {
        // while (record.depth < 0) {
        if (record.done)
            break;

        ray_origin = record.hit_p;
        ray_direction = record.wi;
        ++record.depth;

        optixTrace(optix_launch_params.handle,
                   ray_origin, ray_direction,
                   0.001f, 1e16f, 0.f,
                   255, OPTIX_RAY_FLAG_NONE,
                   0, 2, 0,
                   u0, u1);

        double rr = record.depth > 2 ? 0.95 : 1.0;
        if (record.random.Next() > rr)
            break;
        record.throughput /= rr;
    }

    if (optix_launch_params.frame_cnt > 0) {
        const float t = 1.f / (optix_launch_params.frame_cnt + 1.f);
        const float3 pre = make_float3(optix_launch_params.accum_buffer[pixel_index]);
        record.radiance = lerp(pre, record.radiance, t);
    }
    optix_launch_params.accum_buffer[pixel_index] = make_float4(record.radiance, 1.f);

    float3 color = optix_util::ACESToneMapping(record.radiance, 1.f);
    // color = optix_util::GammaCorrection(color, 2.2f);
    optix_launch_params.frame_buffer[pixel_index] = make_float4(color, 1.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix_util::GetPRD<PathPayloadRecord>();
    // if (optix_launch_params.env) {
    //     // TODO: environment texture
    //     float2 tex = make_float2(0.f, 0.f);
    //     record->radiance += record->throughput * optix_launch_params.env->Sample(tex);
    // }
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
extern "C" __global__ void __closesthit__default() {
    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    auto record = optix_util::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    auto hit_geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    record->hit_p = hit_geo.position;

    if (sbt_data->emitter_index_offset >= 0) {
        unsigned int emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
        auto &emitter = optix_launch_params.emitters[emitter_index];
        auto emission = emitter.radiance.Sample(hit_geo.texcoord);
        if (record->depth == 0) {
            record->radiance += emission;
        } else {
            float distance = length(ray_o - hit_geo.position);
            optix_util::Emitter::LocalRecord emitter_local = emitter.GetLocalInfo(hit_geo.position);
            float LNoL = dot(emitter_local.normal, -ray_dir);
            if (LNoL > 0.f) {
                float light_pdf = distance * distance / (LNoL * emitter.area) * emitter.select_probability;
                float mis = optix_util::MISWeight(record->bsdf_sampled_pdf, light_pdf);
                record->radiance += record->throughput * emission * mis;
            }
        }
    }

    auto &emitter = optix_util::Emitter::SelectOneEmiiter(record->random.Next(), optix_launch_params.emitters);
    auto emitter_local = emitter.SampleDirect(record->random.Next(), record->random.Next());
    float distance = length(emitter_local.position - hit_geo.position);
    float3 L = normalize(emitter_local.position - hit_geo.position);
    float NoL = dot(hit_geo.normal, L);
    float LNoL = dot(emitter_local.normal, -L);

    if (NoL > 0.f && LNoL > 0.f) {
        unsigned int occluded = 0u;
        optixTrace(optix_launch_params.handle,
                   hit_geo.position, L,
                   0.001f, distance - 0.001f, 0.f,
                   255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   1, 2, 1, occluded);
        if (occluded == 0u) {
            float light_pdf = distance * distance / (LNoL * emitter.area) * emitter.select_probability;
            float3 wi = optix_util::ToLocal(L, hit_geo.normal);
            float3 wo = optix_util::ToLocal(-ray_dir, hit_geo.normal);
            auto [f, pdf] = sbt_data->mat.Eval(wi, wo, hit_geo.texcoord);
            if (!optix_util::IsZero(f) && pdf > 0.f) {
                float mis = optix_util::MISWeight(light_pdf, pdf);
                record->radiance += record->throughput * emitter_local.radiance * f * NoL * mis / light_pdf;
            }
        }
    }

    float2 xi = make_float2(record->random.Next(), record->random.Next());
    float3 wo = optix_util::ToLocal(-ray_dir, hit_geo.normal);
    auto [f, wi, pdf] = sbt_data->mat.Sample(xi, wo, hit_geo.texcoord);
    if (optix_util::IsZero(f * abs(wi.z)) || optix_util::IsZero(pdf))
        record->done = true;
    else {
        record->throughput *= f * abs(wi.z) / pdf;
        record->bsdf_sampled_pdf = pdf;
        record->wi = optix_util::ToWorld(wi, hit_geo.normal);
    }
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}