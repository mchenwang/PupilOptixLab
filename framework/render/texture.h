#pragma once

#include "cuda/vec_math.h"
#include <cuda_runtime.h>

namespace Pupil::optix {
    struct Texture {
        enum { Bitmap,
               RGB,
               Checkerboard } type;

        union {
            cudaTextureObject_t bitmap CONST_STATIC_INIT(0);

            float3 rgb;
            struct {
                float3 patch1;
                float3 patch2;
            };
        };

        struct {
            float3 r0;
            float3 r1;
        } transform;

#ifndef PUPIL_CPP
        CUDA_DEVICE float3 Sample(float2 texcoord) const noexcept {
            const auto tex   = make_float3(texcoord, 1.f);
            float      tex_x = dot(transform.r0, tex);
            float      tex_y = dot(transform.r1, tex);
            float3     color;
            switch (type) {
                case RGB:
                    color = rgb;
                    break;
                case Checkerboard: {
                    tex_x = tex_x - (tex_x > 0.f ? floorf(tex_x) : ceilf(tex_x));
                    tex_y = tex_y - (tex_y > 0.f ? floorf(tex_y) : ceilf(tex_y));
                    if (tex_x < 0.f) tex_x += 1.f;
                    if (tex_y < 0.f) tex_y += 1.f;
                    if (tex_x > 0.5f)
                        color = tex_y > 0.5f ? patch1 : patch2;
                    else
                        color = tex_y > 0.5f ? patch2 : patch1;
                } break;
                case Bitmap:
                    color = make_float3(tex2D<float4>(bitmap, tex_x, tex_y));
                    break;
            }
            return color;
        }
#else
        CUDA_DEVICE float3 Sample(float2) const noexcept { return make_float3(0.f); }
#endif
    };
}// namespace Pupil::optix