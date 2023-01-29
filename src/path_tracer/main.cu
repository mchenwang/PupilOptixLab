#include <optix.h>
#include "type.h"

#include "optix_util/util.h"
#include "cuda_util/random.h"

extern "C" {
__constant__ OptixLaunchParams optix_launch_params;
}

struct PathPayloadRecord {
    float3 radiance;
    cuda::Random random;
};

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    auto &camera = optix_launch_params.camera;

    // optix_launch_params.frame_buffer[pixel_index] =
    //     make_float4(
    //         (float)index.x / w,
    //         (float)index.y / h, 0.f, 1.f);

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix_util::PackPointer(&record, u0, u1);

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

    //printf("================%f\n", dot(camera.sample_to_camera.r3, point_on_film));

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
    auto prd = optix_util::GetPRD<PathPayloadRecord>();
    float3 vertices[3];
    optixGetTriangleVertexData(
        optixGetGASTraversableHandle(),
        optixGetPrimitiveIndex(),
        optixGetSbtGASIndex(),
        optixGetRayTime(),
        vertices);

    const float3 v0 = vertices[0];
    const float3 v1 = vertices[1];
    const float3 v2 = vertices[2];
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));
    const float3 ray_dir = optixGetWorldRayDirection();

    const float3 N = faceforward(N_0, -ray_dir, N_0);

    prd->radiance = N * 0.5f + 0.5f;
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