#pragma once

#include <optix.h>
#include "cuda/stream.h"

namespace Pupil::optix {
    class Denoiser {
    public:
        enum EMode : unsigned int {
            None         = 0,
            UseAlbedo    = 1,
            UseNormal    = 1 << 1,
            ApplyToAOV   = 1 << 2,
            UseTemporal  = 1 << 3,
            UseUpscale2X = 1 << 4,
            Tiled        = 1 << 5
        };

        Denoiser(util::CountableRef<cuda::Stream>,
                 unsigned int mode = EMode::UseAlbedo | EMode::UseNormal) noexcept;
        ~Denoiser() noexcept;

        void SetMode(unsigned int mode) noexcept;
        void Setup(unsigned int w, unsigned int h) noexcept;

        void Destroy() noexcept;

        void SetTile(unsigned int tile_w, unsigned int tile_h) noexcept;

        struct ExecutionData {
            CUdeviceptr input         = 0;
            CUdeviceptr output        = 0;
            CUdeviceptr prev_output   = 0;
            CUdeviceptr albedo        = 0;
            CUdeviceptr normal        = 0;
            CUdeviceptr motion_vector = 0;
        };
        void Execute(const ExecutionData&) noexcept;

        operator OptixDenoiser() const noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::optix