#pragma once

#include "cuda/util.h"

namespace Pupil::optix {
    struct Camera {
        mat4x4 sample_to_camera;
        mat4x4 camera_to_world;
    };

    // Camera with proj_view matrix.
    // It can calculate coordinates in NDC space.
    struct LCamera {
        mat4x4 sample_to_camera;
        mat4x4 camera_to_world;
        mat4x4 proj_view;
    };

    // Camera with prev_proj_view(previous frame).
    // It can calculate coordinates at previous frame.
    struct TCamera {
        mat4x4 sample_to_camera;
        mat4x4 camera_to_world;
        mat4x4 prev_proj_view;
    };

    // Camera with proj_view and prev_proj_view(previous frame).
    struct TLCamera {
        mat4x4 sample_to_camera;
        mat4x4 camera_to_world;
        mat4x4 proj_view;
        mat4x4 prev_proj_view;
    };
}// namespace Pupil::optix