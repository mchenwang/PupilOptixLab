#pragma once

#include "cuda/matrix.h"
#include "util/camera.h"

namespace Pupil::optix {
struct Camera {
    mat4x4 sample_to_camera;
    mat4x4 camera_to_world;
};
}// namespace Pupil::optix