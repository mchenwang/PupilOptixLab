#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"
#include "material/optix_material.h"
#include "cuda/matrix.h"

namespace Pupil::pt {
struct OptixLaunchParams {
    struct {
        unsigned int max_depth;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;
    unsigned int sample_cnt;

    cuda::ConstDataView<optix::Camera> camera;
    mat4x4 camera_proj_mat;
    mat4x4 camera_view_mat;
    mat4x4 camera_proj_view_mat;
    optix::EmitterGroup emitters;

    //cuda::RWArrayView<float4> accum_buffer;
    cuda::RWArrayView<float4> frame_buffer;
    cuda::RWArrayView<float4> albedo;
    cuda::RWArrayView<float4> depth;
    cuda::RWArrayView<float4> normal;

    OptixTraversableHandle handle;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

}// namespace Pupil::pt