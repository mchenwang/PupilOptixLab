#pragma once

#include <optix.h>
#include "cuda/stream.h"

namespace Pupil::optix {
class Denoiser {
public:
    enum EMode : unsigned int {
        None = 0,
        UseAlbedo = 1,
        UseNormal = 1 << 1,
        ApplyToAOV = 1 << 2,
        UseTemporal = 1 << 3,
        UseUpscale2X = 1 << 4,
        Tiled = 1 << 5
    };

    Denoiser(unsigned int mode = EMode::UseAlbedo | EMode::UseNormal,
             cuda::Stream *stream = nullptr) noexcept;
    ~Denoiser() noexcept;

    void SetMode(unsigned int mode) noexcept;
    void Setup(unsigned int w, unsigned int h) noexcept;

    void Destroy() noexcept;

    void SetTile(unsigned int tile_w, unsigned int tile_h) noexcept;

    struct ExecutionData {
        CUdeviceptr input = 0;
        CUdeviceptr output = 0;
        CUdeviceptr prev_output = 0;
        CUdeviceptr albedo = 0;
        CUdeviceptr normal = 0;
        CUdeviceptr motion_vector = 0;
    };
    void Execute(const ExecutionData &) noexcept;

    operator OptixDenoiser() const noexcept { return m_denoiser; }

public:
    unsigned int mode = EMode::UseAlbedo | EMode::UseNormal;

    unsigned int input_w = 0;
    unsigned int input_h = 0;
    unsigned int tile_w = 100;
    unsigned int tile_h = 100;

private:
    cuda::Stream *m_stream = nullptr;
    OptixDenoiser m_denoiser = nullptr;
    OptixDenoiserParams m_params{};
    OptixDenoiserGuideLayer m_guide_layer{};

    CUdeviceptr m_scratch = 0;
    size_t m_scratch_size = 0;
    unsigned int m_overlap = 0;

    CUdeviceptr m_state = 0;
    size_t m_state_size = 0;

    CUdeviceptr m_hdr_intensity = 0;
    CUdeviceptr m_hdr_average_color = 0;
};
}// namespace Pupil::optix