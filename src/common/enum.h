#pragma once
#include "macro_map.h"

#define PUPIL_ENUM_NAME(e) _##e
#define PUPIL_ENUM_STR_NAME(e) #e

#define PUPIL_MACRO_ARGS_NUM(...) \
    std::tuple_size<decltype(std::make_tuple(MAP_LIST(PUPIL_ENUM_STR_NAME, __VA_ARGS__)))>::value

#define PUPIL_ENUM_DEFINE(ENUM_CLASS_NAME, ...) \
    enum class ENUM_CLASS_NAME : unsigned int { \
        UNKNOWN = 0,                            \
        MAP_LIST(PUPIL_ENUM_NAME, __VA_ARGS__), \
        COUNT                                   \
    };
#define PUPIL_ENUM_STRING_ARRAY(ENUM_STRING_ARRAY_NAME, ...)         \
    const std::array<std::string, PUPIL_MACRO_ARGS_NUM(__VA_ARGS__)> \
        ENUM_STRING_ARRAY_NAME = { MAP_LIST(PUPIL_ENUM_STR_NAME, __VA_ARGS__) };
