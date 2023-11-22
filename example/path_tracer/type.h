#pragma once

#include "render/camera.h"
#include "render/emitter.h"
#include "render/material.h"
#include "render/geometry.h"

#include "cuda/util.h"

namespace Pupil::pt {
    struct OptixLaunchParams {
        struct {
            unsigned int max_depth;
            bool         accumulated_flag;

            struct {
                unsigned int width;
                unsigned int height;
            } frame;
        } config;
        unsigned int random_seed;
        unsigned int sample_cnt;

        optix::Camera       camera;
        optix::EmitterGroup emitters;

        cuda::RWArrayView<float4> accum_buffer;
        cuda::RWArrayView<float4> frame_buffer;

        cuda::RWArrayView<float4> normal_buffer;
        cuda::RWArrayView<float4> albedo_buffer;
        // cuda::RWArrayView<float>  test;

        OptixTraversableHandle handle;
    };

    struct HitGroupData {
        optix::Material mat;
        optix::Geometry geo;
        int             emitter_index = -1;
    };

}// namespace Pupil::pt