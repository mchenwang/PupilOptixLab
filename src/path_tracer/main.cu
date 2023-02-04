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

    float throughput;

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
    record.throughput = 1.f;
    record.radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.frame_cnt);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());
    // The center of each pixel is at fraction (0.5,0.5)
    // const float2 subpixel =
    //     2.0f * make_float2(
    //                (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
    //                (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h)) -
    //     1.0f;
    const float2 subpixel = make_float2((static_cast<float>(index.x)) / w, (static_cast<float>(index.y)) / h);
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

    // for (uint32_t depth = 0u; depth < optix_launch_params.config.max_depth; ++depth) {
    //     if (record.done)
    //         break;

    //     double rr = depth > 2 ? 0.95 : 1.0;
    //     if (record.random.Next() > rr)
    //         break;

    //     record.throughput /= rr;
    // }

    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.01f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __miss__shadow() {
}
extern "C" __global__ void __closesthit__default() {
    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    auto prd = optix_util::GetPRD<PathPayloadRecord>();
    const auto hit_geo = sbt_data->geo.GetHitLocalGeometry();

    prd->radiance = sbt_data->mat.GetColor(hit_geo.texcoord);
    // printf("%f %f %f\n", prd->radiance.x, prd->radiance.y, prd->radiance.z);
    // prd->radiance = make_float3(hit_geo.texcoord, 0.f);
}
extern "C" __global__ void __closesthit__shadow() {
}
extern "C" __global__ void __closesthit__default_sphere() {
    // PathPayloadRecord *prd = Util::GetPRD<PathPayloadRecord>();
    // prd->hit.ray_t = optixGetRayTmax();

    // const float3 ray_dir = optixGetWorldRayDirection();
    // const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    // float3 N = normalize(P - sbt_data->sphere.center);
}
extern "C" __global__ void __closesthit__shadow_sphere() {
}