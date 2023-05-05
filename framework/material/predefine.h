#pragma once

#include "cuda/preprocessor.h"
#include "util/enum.h"

#include <array>

namespace Pupil::material {
/// Add a new material
/// 1. add name to PUPIL_RENDER_MATERIAL
/// 2. define the material struct
/// 3. add the new material to Material::union struct
/// 4. add the corresponding optix_material
/// 5. implement material loader and [] method for optix
// #define PUPIL_RENDER_MATERIAL \
//     diffuse, dielectric, conductor, roughconductor, twosided, plastic, roughplastic

// PUPIL_ENUM_DEFINE(EMatType, PUPIL_RENDER_MATERIAL)
// PUPIL_ENUM_STRING_ARRAY(S_MAT_TYPE_NAME, PUPIL_RENDER_MATERIAL)

#define PUPIL_MAT_SAMPLE_CALL(mat) __direct_callable__##mat##_sample

enum class EMatType : unsigned int {
    Unknown = 0,
#define PUPIL_MATERIAL_TYPE_DEFINE(type) type,
#include "material_decl.inl"
#undef PUPIL_MATERIAL_TYPE_DEFINE
    Twosided,
    Count
};

const auto S_MAT_TYPE_NAME = std::array{
#define PUPIL_MATERIAL_NAME_DEFINE(name) #name,
#include "material_decl.inl"
#undef PUPIL_MATERIAL_NAME_DEFINE
    "twosided"
};

}// namespace Pupil::material