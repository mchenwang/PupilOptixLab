#pragma once

#include "cuda/preprocessor.h"
#include "util/enum.h"

#include <array>

namespace Pupil {
/// Add a new material
/// 1. declare material name in decl/material_decl.inl
/// 2. xml obj to `material` struct
/// 3. `material` struct to `optix_material` struct
/// 4. implemente bsdf

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
    "twosided"
};

#define PUPIL_MAT_SAMPLE_CALL(mat) __direct_callable__##mat##_sample
#define PUPIL_MAT_EVAL_CALL(mat) __direct_callable__##mat##_eval

}// namespace Pupil