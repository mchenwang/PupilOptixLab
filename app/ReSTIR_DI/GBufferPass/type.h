#pragma once

#include "optix/geometry.h"
#include "optix/scene/emitter.h"
#include "optix/scene/camera.h"
#include "material/optix_material.h"
#include "cuda/data_view.h"
#include "cuda/matrix.h"

#include "../reservoir.h"

struct GBufferPassLaunchParams {
    struct {
        unsigned int width;
        unsigned int height;
    } frame;
    unsigned int random_seed;

    struct {
        mat4x4 sample_to_camera;
        mat4x4 camera_to_world;
        mat4x4 view;
        mat4x4 proj_view;
    } camera;
    // Pupil::cuda::ConstDataView<Pupil::optix::Camera> camera;

    OptixTraversableHandle handle;
    Pupil::optix::EmitterGroup emitters;

    Pupil::cuda::RWArrayView<float4> position;
    Pupil::cuda::RWArrayView<float4> normal;// xyz: normal; w: linear depth
    Pupil::cuda::RWArrayView<float4> albedo;

    Pupil::cuda::RWArrayView<Reservoir> reservoirs;// memset with 0 on cpu
};

struct GBufferPassRayGenData {};
struct GBufferPassMissData {};
struct GBufferPassHitGroupData {
    Pupil::optix::material::Material mat;
    Pupil::optix::Geometry geo;
    int emitter_index_offset = -1;
};
