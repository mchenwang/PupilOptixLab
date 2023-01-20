#pragma once

#include "common/texture.h"

#include "preprocessor.h"
#include <cuda_runtime.h>

namespace cuda {
struct Texture {
    util::ETextureType type = util::ETextureType::RGB;
    union {
        cudaTextureObject_t bitmap = 0;
        float3 rgb;
        struct {
            float3 patch1;
            float3 patch2;
        };
    };

    CUDA_HOSTDEVICE Texture() noexcept {}
};
}// namespace cuda