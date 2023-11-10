#include "util.h"

namespace Pupil::util {
    std::vector<std::string> Split(std::string_view str, std::string_view deli) noexcept {
        std::vector<std::string> ret;
        size_t                   pos_start = 0, pos_end;

        while ((pos_end = str.find(deli, pos_start)) != std::string::npos) {
            ret.emplace_back(str.substr(pos_start, pos_end - pos_start));
            pos_start = pos_end + deli.size();
        }

        ret.emplace_back(str.substr(pos_start));
        return ret;
    }

    bool LoadBool(std::string_view value, bool default_value) noexcept {
        if (value == "true" || value == "1" || value == "True")
            return true;
        else if (value == "false" || value == "0" || value == "False")
            return false;
        return default_value;
    }

    int LoadInt(std::string_view value, int default_value) noexcept {
        if (value.empty())
            return default_value;

        int ret;
        try {
            ret = std::stoi(value.data());
        } catch (...) {
            ret = default_value;
        }
        return ret;
    }

    float LoadFloat(std::string_view value, float default_value) noexcept {
        if (value.empty())
            return default_value;

        float ret;
        try {
            ret = std::stof(value.data());
        } catch (...) {
            ret = default_value;
        }
        return ret;
    }

    util::Float3 LoadFloat3(std::string_view value, util::Float3 default_value) noexcept {
        if (value.empty())
            return default_value;

        auto         xyz = Split(value, ",");
        util::Float3 ret = default_value;
        try {
            if (xyz.size() == 3) {
                ret.x = std::stof(xyz[0]);
                ret.y = std::stof(xyz[1]);
                ret.z = std::stof(xyz[2]);
            } else if (xyz.size() == 1) {
                ret.x = ret.y = ret.z = std::stof(xyz[0]);
            }
        } catch (...) {
            ret = default_value;
        }
        return ret;
    }

    util::Float3 Load3Float(std::string_view value, util::Float3 default_value) noexcept {
        if (value.empty())
            return default_value;

        auto         xyz = Split(value, ",");
        util::Float3 ret = default_value;
        try {
            if (xyz.size() == 3) {
                ret.x = std::stof(xyz[0]);
                ret.y = std::stof(xyz[1]);
                ret.z = std::stof(xyz[2]);
            }
        } catch (...) {
            ret = default_value;
        }
        return ret;
    }

    std::vector<int> LoadIntVector(std::string_view value) noexcept {
        if (value.empty())
            return {};

        auto             elements = Split(value, " ");
        std::vector<int> vector(elements.size());
        try {
            for (int i = 0; i < vector.size(); ++i)
                vector[i] = std::stoi(elements[i]);
        } catch (...) {
        }

        return vector;
    }

    std::vector<float> LoadFloatVector(std::string_view value) noexcept {
        if (value.empty())
            return {};

        auto               elements = Split(value, " ");
        std::vector<float> vector(elements.size());
        try {
            for (int i = 0; i < vector.size(); ++i)
                vector[i] = std::stof(elements[i]);
        } catch (...) {
        }

        return vector;
    }
}// namespace Pupil::util