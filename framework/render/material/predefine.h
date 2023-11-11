#pragma once

#include "cuda/preprocessor.h"

#include <array>

namespace Pupil {
    enum class EMatType : unsigned int {
        Unknown = 0,
#define PUPIL_MATERIAL_TYPE_DEFINE(type) type,
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_DEFINE
        Twosided,
        Count
    };

    const auto S_MAT_TYPE_NAME = std::array{
#define PUPIL_MATERIAL_NAME_DEFINE(name) #name,
#include "decl/material_decl.inl"
#undef PUPIL_MATERIAL_NAME_DEFINE
        "twosided"};

}// namespace Pupil