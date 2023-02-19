#include "util.h"

namespace util {
std::vector<std::string> Split(std::string_view str, std::string_view deli) {
    std::vector<std::string> ret;
    size_t pos_start = 0, pos_end;

    while ((pos_end = str.find(deli, pos_start)) != std::string::npos) {
        ret.emplace_back(str.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end + deli.size();
    }

    ret.emplace_back(str.substr(pos_start));
    return ret;
}

Float3 StrToFloat3(std::string_view str) {
    if (str.empty()) return Float3{ 0.f };

    auto values = Split(str, ",");

    if (values.size() == 1) {
        return Float3{ std::stof(values[0]) };
    } else if (values.size() == 3) {
        return Float3{ std::stof(values[0]), std::stof(values[1]), std::stof(values[2]) };
    }

    return Float3{ 0.f };
}
}// namespace util