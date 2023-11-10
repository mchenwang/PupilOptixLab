#pragma once

#include "resource/shape.h"

#include "hair/cemyuksel_hair.h"

namespace Pupil::resource {
    struct ObjMesh {
        std::vector<float>    vertex;
        std::vector<float>    normal;
        std::vector<float>    texcoord;
        std::vector<uint32_t> index;

        static bool Load(const char* path, ObjMesh& mesh, EShapeLoadFlag flags) noexcept;
    };
}// namespace Pupil::resource