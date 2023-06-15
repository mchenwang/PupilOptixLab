#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"
#include "material/optix_material.h"

namespace Pupil::pt {

struct vMF {
    float3 mu;
    float kappa;
    float mean_cosine;
    float weight_sum;
    float iteration_cnt;
};

struct OptixLaunchParams {
    struct {
        unsigned int max_depth;
        bool accumulated_flag;
        bool use_path_guiding;

        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;
    unsigned int sample_cnt;
    bool update_pass;

    cuda::ConstDataView<optix::Camera> camera;
    mat4x4 pre_proj_view_mat;
    optix::EmitterGroup emitters;

    cuda::RWArrayView<float4> accum_buffer;
    cuda::RWArrayView<float4> frame_buffer;

    // path guiding related buffer
    cuda::RWArrayView<vMF> pre_model_buffer;
    cuda::RWArrayView<vMF> new_model_buffer;
    cuda::RWArrayView<float3> position_buffer;
    cuda::RWArrayView<float3> target_buffer;
    cuda::RWArrayView<float> pdf_buffer;
    cuda::RWArrayView<float> radiance_buffer;
    cuda::RWArrayView<float3> normal_buffer;
    cuda::RWArrayView<float> depth_buffer;

    OptixTraversableHandle handle;
};

struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

}// namespace Pupil::pt
