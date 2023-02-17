#pragma once

#include <string>
#include <charconv>

namespace material {
struct IOREntry {
    const char *name;
    float value;
};

// clang-format off
/**
* Many values are taken from Hecht, Optics,
* Fourth Edition.
*
* The IOR values are from measurements between
* 0 and 20 degrees celsius at ~589 nm.
*/
const static IOREntry S_IOR_DATA[] = {
    { "vacuum",                1.0f },
    { "helium",                1.000036f },
    { "hydrogen",              1.000132f },
    { "air",                   1.000277f },
    { "carbon dioxide",        1.00045f },
    //////////////////////////////////////
    { "water",                 1.3330f },
    { "acetone",               1.36f },
    { "ethanol",               1.361f },
    { "carbon tetrachloride",  1.461f },
    { "glycerol",              1.4729f },
    { "benzene",               1.501f },
    { "silicone oil",          1.52045f },
    { "bromine",               1.661f },
    //////////////////////////////////////
    { "water ice",             1.31f },
    { "fused quartz",          1.458f },
    { "pyrex",                 1.470f },
    { "acrylic glass",         1.49f },
    { "polypropylene",         1.49f },
    { "bk7",                   1.5046f },
    { "sodium chloride",       1.544f },
    { "amber",                 1.55f },
    { "pet",                   1.5750f },
    { "diamond",               2.419f },

    { nullptr,                 0.0f }
};
// clang-format on

static float LoadIor(std::string_view str, float default_value) noexcept {
    if (str.empty()) return default_value;

    float value;
    auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (ec == std::errc() && p == str.data() + str.size())
        return value;

    for (auto &[name, ior] : S_IOR_DATA) {
        if (name != nullptr && str.compare(name) == 0) {
            return ior;
        }
    }

    return default_value;
}

}// namespace material