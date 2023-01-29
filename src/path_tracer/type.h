#pragma once

#include <optix.h>
#include <vector_types.h>

#include "material/optix_material.h"
#include "cuda_util/camera.h"

struct OptixLaunchParams {
    struct {
        unsigned int max_depth;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;

    unsigned int frame_cnt;

    cuda::Camera camera;

    float4 *accum_buffer;
    float4 *frame_buffer;

    OptixTraversableHandle handle;
};

struct RayGenData {
};

struct MissData {
};

struct HitGroupData {
    optix_wrap::material::Material mat;
};