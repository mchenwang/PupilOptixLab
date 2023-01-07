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
}// namespace util