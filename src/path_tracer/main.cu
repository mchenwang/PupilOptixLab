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

    unsigned int depth;
    bool done;
};

struct ShadingData {
    optix_util::LocalGeometry hit_geo;
    float3 ray_dir;

    int emitter_index_offset;
};

static __device__ void Shading(
    PathPayloadRecord *record,
    const ShadingData &data) {

    if (record->depth == 0) {
        if (data.emitter_index_offset > 0) {
            unsigned int emitter_index = data.emitter_index_offset + optixGetPrimitiveIndex();
            record->radiance += optix_launch_params.emitters[emitter_index].radiance.Sample(data.hit_geo.texcoord);
        }
    }

    record->hit_p = data.hit_geo.position;

    // direct illumination sampling
    auto &emitter = optix_util::Emitter::SelectOneEmiiter(record->random.Next(), optix_launch_params.emitters);
    auto emitter_sample = emitter.SampleDirect(record->random.Next(), record->random.Next());
    const float distance = length(emitter_sample.position - data.hit_geo.position);
    const float3 L = normalize(emitter_sample.position - data.hit_geo.position);
    const float NoL = dot(data.hit_geo.normal, L);
    const float LNoL = dot(emitter_sample.normal, -L);

    record->wi = L;

    if (NoL > 0.f && LNoL > 0.f) {
        unsigned int occluded = 0u;
        optixTrace(optix_launch_params.handle,
                   data.hit_geo.position, L,
                   0.001f, distance - 0.001f, 0.f,
                   255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   1, 2, 1,
                   occluded);

        if (!occluded) {
            float light_pdf = distance * distance / (LNoL * emitter.area) * emitter.select_probability;
            float f = M_1_PIf;

            record->radiance += record->throughput * emitter_sample.radiance * f * NoL / light_pdf;
        }
    }
}

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

    // optixTrace(optix_launch_params.handle,
    //            ray_origin, ray_direction,
    //            0.001f, 1e16f, 0.f,
    //            255, OPTIX_RAY_FLAG_NONE,
    //            0, 2, 0,
    //            u0, u1);

    for (; record.depth < optix_launch_params.config.max_depth; ++record.depth) {
        if (record.done)
            break;

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

        ray_origin = record.hit_p;
        ray_direction = record.wi;
    }

    optix_launch_params.frame_buffer[pixel_index] = make_float4(record.radiance, 1.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix_util::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.env) {
        // TODO: environment texture
        float2 tex = make_float2(0.f, 0.f);
        record->radiance += record->throughput * optix_launch_params.env->Sample(tex);
    }
    record->done = true;
}
extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
extern "C" __global__ void __closesthit__default() {
    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    auto record = optix_util::GetPRD<PathPayloadRecord>();

    ShadingData shading_data;
    shading_data.emitter_index_offset = sbt_data->emitter_index_offset;
    shading_data.hit_geo = sbt_data->geo.GetHitLocalGeometry();
    shading_data.ray_dir = optixGetWorldRayDirection();

    Shading(record, shading_data);
    record->throughput *= sbt_data->mat.GetColor(shading_data.hit_geo.texcoord);
    // record->done = true;
    // record->radiance = sbt_data->mat.GetColor(hit_geo.texcoord);
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}
extern "C" __global__ void __closesthit__default_sphere() {
    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    auto record = optix_util::GetPRD<PathPayloadRecord>();

    ShadingData shading_data;
    shading_data.emitter_index_offset = sbt_data->emitter_index_offset;
    shading_data.hit_geo = sbt_data->geo.GetHitLocalGeometry();
    shading_data.ray_dir = optixGetWorldRayDirection();

    Shading(record, shading_data);
}
extern "C" __global__ void __closesthit__shadow_sphere() {
    optixSetPayload_0(1u);
}