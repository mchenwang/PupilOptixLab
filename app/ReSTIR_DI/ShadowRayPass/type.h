#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "material/optix_material.h"
#include "cuda/data_view.h"

#include "../reservoir.h"

struct ShadowRayPassLaunchParams {
    struct {
        unsigned int width;
        unsigned int height;
    } frame;

    unsigned int type;
    OptixTraversableHandle handle;
    Pupil::cuda::ConstArrayView<float4> position;
    Pupil::cuda::ConstArrayView<float4> normal;
    Pupil::cuda::ConstArrayView<float4> albedo;

    Pupil::cuda::RWArrayView<Reservoir> reservoirs;

    Pupil::cuda::RWArrayView<float4> frame_buffer;
};

struct ShadowRayPassRayGenData {};
struct ShadowRayPassMissData {};
struct ShadowRayPassHitGroupData {};
