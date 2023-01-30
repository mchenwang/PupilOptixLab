#pragma once

#include "common/texture.h"

#include "preprocessor.h"
#include "cuda_util/vec_math.h"
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

    struct {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 r3;
    } transform;

    CUDA_HOSTDEVICE Texture() noexcept {}

#if defined(__CUDACC__) || defined(__CUDABE__)
    CUDA_HOSTDEVICE float3 Sample(float2 texcoord) const noexcept {
        const float4 tex = make_float4(texcoord, 0.f, 1.f);
        float tex_x = dot(transform.r0, tex);
        float tex_y = dot(transform.r1, tex);
        float3 color;
        switch (type) {
            case util::ETextureType::RGB:
                color = rgb;
                break;
            case util::ETextureType::Checkerboard: {
                tex_x = tex_x - (tex_x > 0.f ? floorf(tex_x) : ceilf(tex_x));
                tex_y = tex_y - (tex_y > 0.f ? floorf(tex_y) : ceilf(tex_y));
                if (tex_x < 0.f) tex_x += 1.f;
                if (tex_y < 0.f) tex_y += 1.f;
                if (tex_x > 0.5f)
                    color = tex_y > 0.5f ? patch1 : patch2;
                else
                    color = tex_y > 0.5f ? patch2 : patch1;
            } break;
            case util::ETextureType::Bitmap:
                color = make_float3(tex2D<float4>(bitmap, tex_x, tex_y));
                break;
        }
        return color;
    }
#endif
};
}// namespace cuda