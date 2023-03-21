#pragma once

#include <optix.h>
#include <vector_types.h>

#include "material/optix_material.h"
#include "optix_util/geometry.h"
#include "optix_util/camera.h"
#include "optix_util/emitter.h"

#include "cuda_util/data_view.h"

struct GBufferPassOptixLaunchParams {
    struct {
        unsigned int width;
        unsigned int height;
    } frame;

    unsigned int frame_cnt;

    cuda::ConstDataView<optix_util::Camera> camera;

    float4 *albedo_buffer;
    float4 *normal_buffer;

    OptixTraversableHandle handle;
};

struct GBufferPassRayGenData {
};

struct GBufferPassMissData {
};

struct GBufferPassHitGroupData {
    optix_util::Geometry geo;
};