#include <optix.h>
#include "GBufferPass/type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ GBufferPassLaunchParams optix_launch_params;
}

struct PathPayloadRecord {
    cuda::Random random;
    unsigned int pixel_index;
    bool hit_flag;
    bool is_emitter;

    float3 color;
    float3 pos;
    float3 normal;
    float depth;
};

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.frame.width;
    const unsigned int h = optix_launch_params.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    auto &camera = optix_launch_params.camera;

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);
    record.color = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);
    record.pixel_index = pixel_index;
    record.hit_flag = false;
    record.is_emitter = false;
    record.depth = 1e16f;

    const float2 subpixel_jitter = make_float2(0.5f);
    // const float2 subpixel_jitter = record.random.Next2();
    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
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

    optix_launch_params.reservoirs[pixel_index].Init();

    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    optix_launch_params.reservoirs[pixel_index].CalcW();

    optix_launch_params.position[pixel_index] = make_float4(record.pos, record.hit_flag ? 1.f : 0.f);
    optix_launch_params.albedo[pixel_index] = make_float4(record.color, record.is_emitter ? 1.f : 0.f);
    optix_launch_params.normal[pixel_index] = make_float4(record.normal, record.depth);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {
    const GBufferPassHitGroupData *sbt_data = (GBufferPassHitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();

    auto geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    record->color = sbt_data->mat.GetColor(geo.texcoord);
    record->normal = geo.normal;
    record->pos = geo.position;
    // +Z points -view
    record->depth = -(optix_launch_params.camera.view * make_float4(geo.position, 1.f)).z;
    record->hit_flag = true;

    if (sbt_data->emitter_index_offset >= 0) {
        record->is_emitter = true;
        auto emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
        auto &emitter = optix_launch_params.emitters.areas[emitter_index];
        auto emission = emitter.GetRadiance(geo.texcoord);

        record->color = emission;
        return;
    }

    for (unsigned int i = 0u; i < 32; i++) {
        auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(record->random.Next());
        auto emitter_sample_record = emitter.SampleDirect(geo, record->random.Next2());

        Reservoir::Sample x_i;
        x_i.pos = emitter_sample_record.pos;
        x_i.distance = emitter_sample_record.distance;
        x_i.normal = emitter_sample_record.normal;
        x_i.emission = emitter_sample_record.radiance;
        x_i.radiance = make_float3(0.f);
        float w_i = 0.f;

        float3 wi = optix::ToLocal(emitter_sample_record.wi, geo.normal);
        float3 wo = optix::ToLocal(-ray_dir, geo.normal);
        // auto [f, pdf] = sbt_data->mat.Eval(wi, wo, geo.texcoord);
        float3 f = make_float3(0.f);
        if (wi.z > 0.f && wo.z > 0.f) {
            f = record->color * M_1_PIf;
        }
        if (!optix::IsZero(f)) {
            float NoL = dot(geo.normal, emitter_sample_record.wi);
            emitter_sample_record.pdf *= emitter.select_probability;
            if (emitter_sample_record.pdf > 0.f) {
                x_i.radiance += emitter_sample_record.radiance * f * NoL;
                w_i = 1.f / emitter_sample_record.pdf;
            }
        }
        x_i.p_hat = optix::GetLuminance(x_i.radiance);
        w_i *= x_i.p_hat;

        optix_launch_params.reservoirs[record->pixel_index].Update(x_i, w_i, record->random);
    }
}