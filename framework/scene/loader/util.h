#pragma once

#include "resource/mi_xml/xml_object.h"
#include "util/util.h"
#include "util/math.h"

namespace Pupil::util {
    std::vector<std::string> Split(std::string_view str, std::string_view deli) noexcept;

    bool               LoadBool(std::string_view value, bool default_value = 0) noexcept;
    int                LoadInt(std::string_view value, int default_value = 0) noexcept;
    float              LoadFloat(std::string_view value, float default_value = 0.f) noexcept;
    Float3             Load3Float(std::string_view value, Float3 default_value = Float3(0.f)) noexcept;
    Float3             LoadFloat3(std::string_view value, Float3 default_value = Float3(0.f)) noexcept;
    std::vector<int>   LoadIntVector(std::string_view value) noexcept;
    std::vector<float> LoadFloatVector(std::string_view value) noexcept;

    static std::vector<std::string> g_supported_scene_format{".xml"};
}// namespace Pupil::util