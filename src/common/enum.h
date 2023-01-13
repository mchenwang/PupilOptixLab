#pragma once
#include "macro_map.h"

#define PUPIL_ENUM_NAME(e) _##e
#define PUPIL_ENUM_STR_NAME(e) #e

#define PUPIL_MACRO_ARGS_NUM(...) \
    std::tuple_size<decltype(std::make_tuple(MAP_LIST(PUPIL_ENUM_STR_NAME, __VA_ARGS__)))>::value

#define PUPIL_ENUM_DEFINE(ENUM_CLASS_NAME, ...) \
    enum class ENUM_CLASS_NAME : unsigned int { \
        _unknown = 0,                           \
        MAP_LIST(PUPIL_ENUM_NAME, __VA_ARGS__), \
        _count                                  \
    };

#define PUPIL_ENUM_STRING_ARRAY(ENUM_STRING_ARRAY_NAME, ...) \
    const auto                                               \
        ENUM_STRING_ARRAY_NAME = std::array{ MAP_LIST(PUPIL_ENUM_STR_NAME, __VA_ARGS__) };
