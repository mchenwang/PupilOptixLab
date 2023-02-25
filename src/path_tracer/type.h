#pragma once

#include <optix.h>
#include <vector_types.h>

#include "material/optix_material.h"
#include "optix_util/geometry.h"
#include "optix_util/camera.h"
#include "optix_util/emitter.h"

#include "cuda_util/data_view.h"

struct OptixLaunchParams {
    struct {
        unsigned int max_depth;
        bool accumulated_flag;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int frame_cnt;

    cuda::ConstDataView<optix_util::Camera> camera;
    cuda::ConstDataView<cuda::Texture> env;
    cuda::ConstArrayView<optix_util::Emitter> emitters;

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
    int emitter_index_offset = -1;
};