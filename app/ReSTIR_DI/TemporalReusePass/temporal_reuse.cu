#include <optix.h>
#include "TemporalReusePass/type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ TemporalReusePassLaunchParams optix_launch_params;
}

extern "C" __global__ void __raygen__main() {
    if (optix_launch_params.random_seed == 1) return;
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.frame.width;
    const unsigned int h = optix_launch_params.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    float4 pos_with_flag = optix_launch_params.position[pixel_index];
    if (pos_with_flag.w <= 0.f) return;
    float3 pos = make_float3(pos_with_flag);

    const auto &camera = optix_launch_params.camera;
    float4 prev_ndc_pos = camera.prev_proj_view * make_float4(pos, 1.f);
    prev_ndc_pos /= prev_ndc_pos.w;
    float2 prev_screen_pos = make_float2(prev_ndc_pos.x, prev_ndc_pos.y) * 0.5f + 0.5f;
    if (prev_screen_pos.x < 0 || prev_screen_pos.x >= 1.f || prev_screen_pos.y < 0 || prev_screen_pos.y >= 1.f)
        return;

    const uint2 prev_pixel = make_uint2(prev_screen_pos.x * w, prev_screen_pos.y * h);
    const unsigned int prev_pixel_index = prev_pixel.y * w + prev_pixel.x;

    float3 prev_pos = make_float3(optix_launch_params.prev_position[prev_pixel_index]);
    if (!optix::IsZero(prev_pos - pos)) return;

    cuda::Random random;
    random.Init(4, pixel_index, optix_launch_params.random_seed);

    auto &prev_reservoirs = optix_launch_params.prev_frame_reservoirs[prev_pixel_index];
    unsigned int prev_reuse_limit_num = 20 * optix_launch_params.reservoirs[pixel_index].M;
    if (prev_reservoirs.M > prev_reuse_limit_num) {
        prev_reservoirs.M = prev_reuse_limit_num;
    }
    optix_launch_params.reservoirs[pixel_index].Combine(prev_reservoirs, random);
    return;
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {
}