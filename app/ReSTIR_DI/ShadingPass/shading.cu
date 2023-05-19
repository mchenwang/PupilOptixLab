#include <optix.h>
#include "ShadingPass/type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ShadingPassLaunchParams optix_launch_params;
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
        if (albedo.w > 0.f) {
            color = make_float3(albedo);
        } else {
            float3 hit_pos = make_float3(pos);
            float3 hit_nor = make_float3(optix_launch_params.normal[pixel_index]);
            auto &reservoir = optix_launch_params.reservoirs[pixel_index];

            // float3 L = normalize(reservoir.y.pos - hit_pos);
            // float dist = length(reservoir.y.pos - hit_pos);
            // float NoL = max(0.f, dot(L, hit_nor));
            if (optix_launch_params.debug_type == 0) {
                color = reservoir.y.radiance * reservoir.W;
            } else if (optix_launch_params.debug_type == 1) {
                color = reservoir.y.radiance;
            } else if (optix_launch_params.debug_type == 2) {
                color = make_float3(1.f, 1.f, 1.f) * reservoir.y.p_hat;
            } else if (optix_launch_params.debug_type == 3) {
                color = make_float3(1.f, 1.f, 1.f) * reservoir.W;
            } else if (optix_launch_params.debug_type == 4) {
                color = make_float3(1.f, 1.f, 1.f) * reservoir.M;
            } else if (optix_launch_params.debug_type == 5) {
                color = make_float3(1.f, 1.f, 1.f) * reservoir.w_sum;
            }
            // color = make_float3(reservoir.M);
            // color = hit_nor * 0.5f + 0.5f;
            // color = reservoir.y.radiance;

            // color = reservoir.y.emission * make_float3(albedo) * reservoir.W * M_1_PIf * NoL / dist / dist;
        }

    } else {
        //TODO env light
    }

    optix_launch_params.frame_buffer[pixel_index] = make_float4(color, 1.f);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {
}