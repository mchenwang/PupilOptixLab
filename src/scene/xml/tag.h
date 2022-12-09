#pragma once

#include <string_view>
#include <array>

#include "common/enum.h"

namespace scene {
namespace xml {

#define PUPIL_XML_TAGS             \
    scene,                         \
        default,                   \
        bsdf,                      \
        emitter,                   \
        film,                      \
        integrator,                \
        sensor,                    \
        shape,                     \
        texture,                   \
        lookat,                    \
        transform, /*xml objects*/ \
        integer,                   \
        string,                    \
        float,                     \
        rgb,                       \
        matrix,                    \
        scale,                     \
        rotate,                    \
        translate,                 \
        boolean, /*properties*/    \
        ref

PUPIL_ENUM_DEFINE(ETag, PUPIL_XML_TAGS)
PUPIL_ENUM_STRING_ARRAY(S_TAGS_NAME, PUPIL_XML_TAGS)

inline std::string TagToString(ETag tag) noexcept {
    auto index = static_cast<unsigned int>(tag);
    if (index < 1 || index >= (unsigned int)ETag::COUNT) return "unknown";
    return S_TAGS_NAME[index - 1];
}
}
}// namespace scene::xml