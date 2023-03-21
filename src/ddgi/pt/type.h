#pragma once

#include <optix.h>
#include <vector_types.h>

#include "material/optix_material.h"
#include "optix_util/geometry.h"
#include "optix_util/camera.h"
#include "optix_util/emitter.h"

#include "cuda_util/data_view.h"

struct PTPassOptixLaunchParams {
    struct {
        unsigned int max_depth;
        bool accumulated_flag;
        bool use_tone_mapping;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int frame_cnt;
    unsigned int sample_cnt;

    cuda::ConstDataView<optix_util::Camera> camera;
    optix_util::EmitterGroup emitters;

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