#pragma once

#include "math.h"

namespace Pupil::util {
    template<typename T>
    class Hash : public std::hash<T> {};

    template<class T>
    inline void HashCombine(std::size_t& s, const T& v) {
        Hash<T> h;
        s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
    }

    template<>
    struct Hash<Float3> {
        size_t operator()(const Float3& xyz) const {
            std::size_t res = 0;
            HashCombine(res, xyz.x);
            HashCombine(res, xyz.y);
            HashCombine(res, xyz.z);
            return res;
        }
    };

    template<>
    struct Hash<Float4> {
        size_t operator()(const Float4& xyzw) const {
            std::size_t res = 0;
            HashCombine(res, xyzw.x);
            HashCombine(res, xyzw.y);
            HashCombine(res, xyzw.z);
            HashCombine(res, xyzw.w);
            return res;
        }
    };
}// namespace Pupil::util