#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "material/optix_material.h"
#include "cuda/data_view.h"

#include "../reservoir.h"

struct TemporalReusePassLaunchParams {
    struct {
        unsigned int width;
        unsigned int height;
    } frame;
    unsigned int random_seed;

    Pupil::cuda::ConstArrayView<float4> position;
    Pupil::cuda::ConstArrayView<float4> prev_position;

    Pupil::cuda::RWArrayView<Reservoir> reservoirs;
    Pupil::cuda::RWArrayView<Reservoir> prev_frame_reservoirs;

    struct {
        mat4x4 prev_proj_view;
    } camera;
};

struct TemporalReusePassRayGenData {};
struct TemporalReusePassMissData {};
struct TemporalReusePassHitGroupData {};
