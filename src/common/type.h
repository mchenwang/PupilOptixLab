#pragma once

namespace util {
struct float3 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
    };
};
}// namespace util