#include <optix.h>
#include "ShadowRayPass/type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ShadowRayPassLaunchParams optix_launch_params;
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.frame.width;
    // const unsigned int h = optix_launch_params.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    const auto pos = optix_launch_params.position[pixel_index];
    float3 color = make_float3(0.f);
    if (pos.w > 0.f) {
        const auto albedo = optix_launch_params.albedo[pixel_index];
        if (albedo.w == 0.f) {
            float3 hit_pos = make_float3(pos);
            auto &reservoir = optix_launch_params.reservoirs[pixel_index];

            float3 L = normalize(reservoir.y.pos - hit_pos);
            float dist = length(reservoir.y.pos - hit_pos);
            unsigned int miss_flag = 0;
            optixTrace(
                optix_launch_params.handle,
                hit_pos, L,
                0.00001f, dist - 0.00001f, 0.f,
                255,
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                0, 2, 0, miss_flag);

            if (!miss_flag) {
                optix_launch_params.reservoirs[pixel_index].Init();
            }
        }
    }
}

extern "C" __global__ void __miss__default() {
    optixSetPayload_0(1u);
}
extern "C" __global__ void __closesthit__default() {
}