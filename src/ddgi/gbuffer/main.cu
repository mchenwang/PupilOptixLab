#include <optix.h>
#include "type.h"

#include "optix_util/util.h"
#include "optix_util/geometry.h"
#include "optix_util/emitter.h"

#include "cuda_util/random.h"

extern "C" {
__constant__ GBufferPassOptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix_util::LocalGeometry geo;
    optix_util::material::Material mat;
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
    const unsigned int w = optix_launch_params.frame.width;
    const unsigned int h = optix_launch_params.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    auto &camera = *optix_launch_params.camera.operator->();

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix_util::PackPointer(&record, u0, u1);

    record.done = false;
    record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.env_radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.frame_cnt);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());

    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = make_float4(
        dot(camera.sample_to_camera.r0, point_on_film),
        dot(camera.sample_to_camera.r1, point_on_film),
        dot(camera.sample_to_camera.r2, point_on_film),
        dot(camera.sample_to_camera.r3, point_on_film));

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = normalize(make_float3(
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

    int depth = 0;
    auto local_hit = record.hit;

    if (!record.done) {
        optix_launch_params.normal_buffer[pixel_index] = make_float4(local_hit.geo.normal * 0.5f + 0.5f, 1.f);
        // TODO:
        // optix_launch_params.albedo_buffer =
    } else {
        optix_launch_params.normal_buffer[pixel_index] = make_float4(0.f);
    }
}

extern "C" __global__ void __miss__default() {
    auto record = optix_util::GetPRD<PathPayloadRecord>();
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
}
extern "C" __global__ void __closesthit__default() {
    const GBufferPassHitGroupData *sbt_data = (GBufferPassHitGroupData *)optixGetSbtDataPointer();
    auto record = optix_util::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    record->hit.geo = sbt_data->geo.GetHitLocalGeometry();
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}