#pragma once

#include <string_view>
#include <array>

#include "common/macro_map.h"

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

#define TAG_ENUM_NAME(tag) _##tag
#define TAG_STR_NAME(tag) #tag

#define TAG_ARGS_NUM(...) std::tuple_size<decltype(std::make_tuple(MAP_LIST(TAG_STR_NAME, PUPIL_XML_TAGS)))>::value

#define TAGS_DEFINE(...)                               \
    enum class ETag : unsigned int {                   \
        UNKNOWN = 0,                                   \
        MAP_LIST(TAG_ENUM_NAME, __VA_ARGS__),          \
        COUNT                                          \
    };                                                 \
    std::array<std::string, TAG_ARGS_NUM(__VA_ARGS__)> \
        S_TAGS_NAME = { MAP_LIST(TAG_STR_NAME, __VA_ARGS__) };

TAGS_DEFINE(PUPIL_XML_TAGS)
}
}// namespace scene::xml