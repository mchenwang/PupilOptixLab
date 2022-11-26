#include <optix.h>
#include "type.h"

extern "C" {
__constant__ OptixLaunchParams optix_launch_params;
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    optix_launch_params.frame_buffer[pixel_index] = make_float4(index.x, index.y, 1.f, 1.f);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __miss__shadow() {
}
extern "C" __global__ void __closesthit__default() {
}
extern "C" __global__ void __closesthit__shadow() {
}
extern "C" __global__ void __closesthit__default_sphere() {
}
extern "C" __global__ void __closesthit__shadow_sphere() {
}