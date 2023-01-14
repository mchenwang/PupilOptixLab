#pragma once

#include "transform.h"
#include "type.h"

namespace util {
enum class ETextureAddressMode : unsigned int {
    Wrap = 0,
    Clamp = 1,
    Mirror = 2,
    Border = 3
};

enum class ETextureFilterMode : unsigned int {
    Point = 0,
    Linear = 1
};

enum class ETextureType : unsigned int {
    RGB = 0,
    Bitmap,
    Checkerboard
};

struct RGBTexture {
    float3 color{ 0.f, 0.f, 0.f };
};

struct CheckerboardTexture {
    float3 patch1{ 0.4f, 0.4f, 0.4f };
    float3 patch2{ 0.2f, 0.2f, 0.2f };
};

struct BitmapTexture {
    size_t w = 0;
    size_t h = 0;
    float *data = nullptr;

    ETextureAddressMode address_mode = ETextureAddressMode::Wrap;
    ETextureFilterMode filter_mode = ETextureFilterMode::Linear;
};

struct Texture {
    ETextureType type = ETextureType::RGB;
    union {
        RGBTexture rgb{};
        BitmapTexture bitmap;
        CheckerboardTexture checkerboard;
    };
    Transform transform{};
};
}// namespace util