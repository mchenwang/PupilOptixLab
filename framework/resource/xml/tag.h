#pragma once

#include <string_view>
#include <array>

#include "util/enum.h"

namespace Pupil::resource {
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
        point,                     \
        matrix,                    \
        scale,                     \
        rotate,                    \
        translate,                 \
        boolean, /*properties*/    \
        ref

PUPIL_ENUM_DEFINE(ETag, PUPIL_XML_TAGS)
PUPIL_ENUM_STRING_ARRAY(S_TAGS_NAME, PUPIL_XML_TAGS)
}
}// namespace Pupil::resource::xml