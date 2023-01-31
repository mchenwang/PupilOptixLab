#pragma once

#include <optix.h>
#include <vector_types.h>

#include "material/optix_material.h"
#include "optix_util/geometry.h"
#include "cuda_util/camera.h"
#include "cuda_util/data_view.h"

struct OptixLaunchParams {
    struct {
        unsigned int max_depth;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;

    unsigned int frame_cnt;

    cuda::ConstDataView<cuda::Camera> camera;

    float4 *accum_buffer;
    float4 *frame_buffer;

    OptixTraversableHandle handle;
};

struct RayGenData {
};

struct MissData {
};

struct HitGroupData {
    optix_util::material::Material mat;
    optix_util::Geometry geo;
};