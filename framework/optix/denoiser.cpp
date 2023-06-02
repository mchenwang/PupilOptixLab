#include "denoiser.h"
#include "context.h"
#include "check.h"
#include "cuda/util.h"
#include "util/log.h"

#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

namespace {
inline bool FillOptixImageStruct(OptixImage2D &ret, CUdeviceptr data, unsigned int w, unsigned h, bool upscale = false) {
    if (!data) [[unlikely]]
        return false;

    ret = {
        .data = data,
        .width = w * (upscale ? 2 : 1),
        .height = h * (upscale ? 2 : 1),
        .rowStrideInBytes = static_cast<unsigned int>(w * (upscale ? 2 : 1) * sizeof(float4)),
        .pixelStrideInBytes = sizeof(float4),
        .format = OPTIX_PIXEL_FORMAT_FLOAT4
    };
    return true;
}
}// namespace

namespace Pupil::optix {
Denoiser::Denoiser(unsigned int mode, cuda::Stream *stream) noexcept {
    m_stream = stream;
    SetMode(mode);
}
Denoiser::~Denoiser() noexcept { Destroy(); }

void Denoiser::Destroy() noexcept {
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    m_denoiser = nullptr;

    CUDA_FREE(m_hdr_intensity);
    CUDA_FREE(m_hdr_average_color);
    CUDA_FREE(m_guide_layer.outputInternalGuideLayer.data);
    CUDA_FREE(m_guide_layer.previousOutputInternalGuideLayer.data);
    CUDA_FREE(m_scratch);
    CUDA_FREE(m_state);
}

void Denoiser::SetMode(unsigned int mode) noexcept {
    if (m_denoiser) {
        if (mode == this->mode)
            return;

        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        m_denoiser = nullptr;
    }

    this->mode = mode;

    OptixDenoiserOptions options{
        .guideAlbedo = mode & EMode::UseAlbedo,
        .guideNormal = mode & EMode::UseNormal
    };

    OptixDenoiserModelKind kind;
    if (mode & EMode::UseUpscale2X) {
        kind = mode & EMode::UseTemporal ?
                   OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X :
                   OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    } else if (mode & EMode::ApplyToAOV) {
        kind = mode & EMode::UseTemporal ?
                   OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV :
                   OPTIX_DENOISER_MODEL_KIND_AOV;
    } else {
        kind = mode & EMode::UseTemporal ?
                   OPTIX_DENOISER_MODEL_KIND_TEMPORAL :
                   OPTIX_DENOISER_MODEL_KIND_HDR;
    }

    auto ctx = util::Singleton<optix::Context>::instance();
    OPTIX_CHECK(optixDenoiserCreate(*ctx, kind, &options, &m_denoiser));
}

void Denoiser::SetTile(unsigned int tile_w, unsigned int tile_h) noexcept {
    if (tile_w > input_w || tile_h > input_h) {
        Log::Warn("tile size({}x{}) must be smaller than "
                  "input film size({}x{}).",
                  tile_w, tile_h, input_w, input_h);
        return;
    }
    this->tile_w = tile_w;
    this->tile_h = tile_h;
}

void Denoiser::Setup(unsigned int w, unsigned int h) noexcept {
    if (w == 0 || h == 0) {
        Log::Error(" [Denoiser Setup] size must be >0.");
        return;
    }
    input_w = w;
    input_h = h;

    auto size_w = mode & EMode::Tiled ? tile_w : w;
    auto size_h = mode & EMode::Tiled ? tile_w : h;

    OptixDenoiserSizes sizes{};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, size_w, size_h, &sizes));

    if (mode & EMode::Tiled) {
        m_scratch_size = sizes.withOverlapScratchSizeInBytes;
        m_overlap = sizes.overlapWindowSizeInPixels;
    } else {
        m_scratch_size = sizes.withoutOverlapScratchSizeInBytes;
        m_overlap = 0;
    }

    if (mode & EMode::UseUpscale2X || mode & EMode::ApplyToAOV) {
        if (m_hdr_average_color == 0)
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_hdr_average_color), 3 * sizeof(float)));
    } else {
        if (m_hdr_intensity == 0)
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_hdr_intensity), sizeof(float)));
    }

    CUDA_FREE(m_scratch);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_scratch), m_scratch_size));

    m_state_size = sizes.stateSizeInBytes;
    CUDA_FREE(m_state);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_state), m_state_size));

    OPTIX_CHECK(optixDenoiserSetup(
        m_denoiser,
        *m_stream,
        mode & EMode::Tiled ? tile_w + 2 * m_overlap : w,
        mode & EMode::Tiled ? tile_h + 2 * m_overlap : h,
        m_state,
        m_state_size,
        m_scratch,
        m_scratch_size));

    m_params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    m_params.hdrIntensity = m_hdr_intensity;
    m_params.hdrAverageColor = m_hdr_average_color;
    m_params.blendFactor = 0.0f;
    m_params.temporalModeUsePreviousLayers = 0;

    if (mode & EMode::UseTemporal) {
        CUdeviceptr internal_memory_in = 0;
        CUdeviceptr internal_memory_out = 0;
        size_t internal_size = (mode & EMode::Tiled ? 4 : 1) *
                               input_h * input_w *
                               sizes.internalGuideLayerPixelSizeInBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&internal_memory_in), internal_size));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&internal_memory_out), internal_size));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(internal_memory_in), 0, internal_size));

        CUDA_FREE(m_guide_layer.previousOutputInternalGuideLayer.data);
        CUDA_FREE(m_guide_layer.outputInternalGuideLayer.data);

        m_guide_layer.previousOutputInternalGuideLayer.data = internal_memory_in;
        m_guide_layer.previousOutputInternalGuideLayer.width = (mode & EMode::Tiled ? 2 : 1) * input_w;
        m_guide_layer.previousOutputInternalGuideLayer.height = (mode & EMode::Tiled ? 2 : 1) * input_h;
        m_guide_layer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned(sizes.internalGuideLayerPixelSizeInBytes);
        m_guide_layer.previousOutputInternalGuideLayer.rowStrideInBytes = m_guide_layer.previousOutputInternalGuideLayer.width *
                                                                          m_guide_layer.previousOutputInternalGuideLayer.pixelStrideInBytes;
        m_guide_layer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

        m_guide_layer.outputInternalGuideLayer = m_guide_layer.previousOutputInternalGuideLayer;
        m_guide_layer.outputInternalGuideLayer.data = internal_memory_out;
    }
}

void Denoiser::Execute(const ExecutionData &data) noexcept {
    if (input_h == 0 || input_w == 0) [[unlikely]] {
        Log::Warn("Denoiser does not setup.");
        return;
    }

    OptixDenoiserLayer layer{};
    if (!FillOptixImageStruct(layer.input, data.input, input_w, input_h)) {
        Log::Error("Denoiser input ptr can not be NULL.");
        return;
    }

    if (!FillOptixImageStruct(layer.output, data.output, input_w, input_h, mode & EMode::UseUpscale2X)) {
        Log::Error("Denoiser output ptr can not be NULL.");
        return;
    }

    if (mode & EMode::UseAlbedo) {
        if (!FillOptixImageStruct(m_guide_layer.albedo, data.albedo, input_w, input_h)) {
            Log::Error("Denoiser input albedo ptr can not be NULL.");
            return;
        }
    }

    if (mode & EMode::UseNormal) {
        if (!FillOptixImageStruct(m_guide_layer.normal, data.normal, input_w, input_h)) {
            Log::Error("Denoiser input normal ptr can not be NULL.");
            return;
        }
    }

    if (mode & EMode::UseTemporal) {
        if (!FillOptixImageStruct(layer.previousOutput, data.prev_output, input_w, input_h, mode & EMode::UseUpscale2X)) {
            Log::Error("Denoiser input prev_output ptr can not be NULL.");
            return;
        }
        if (!FillOptixImageStruct(m_guide_layer.flow, data.motion_vector, input_w, input_h, mode & EMode::UseUpscale2X)) {
            Log::Error("Denoiser input motion_vector ptr can not be NULL.");
            return;
        }
    }

    if (m_hdr_intensity) {
        OPTIX_CHECK(optixDenoiserComputeIntensity(
            m_denoiser,
            *m_stream,
            &layer.input,
            m_hdr_intensity,
            m_scratch,
            m_scratch_size));
    }
    if (m_hdr_average_color) {
        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            m_denoiser,
            *m_stream,
            &layer.input,
            m_hdr_average_color,
            m_scratch,
            m_scratch_size));
    }

    if (mode & EMode::Tiled) {
        OPTIX_CHECK(optixUtilDenoiserInvokeTiled(
            m_denoiser,
            *m_stream,
            &m_params,
            m_state,
            m_state_size,
            &m_guide_layer,
            &layer,
            1,
            m_scratch,
            m_scratch_size,
            m_overlap,
            tile_w,
            tile_h));
    } else {
        OPTIX_CHECK(optixDenoiserInvoke(
            m_denoiser,
            *m_stream,
            &m_params,
            m_state,
            m_state_size,
            &m_guide_layer,
            &layer,
            1,
            0,// input offset X
            0,// input offset y
            m_scratch,
            m_scratch_size));
    }

    if (mode & EMode::UseTemporal) {
        std::swap(m_guide_layer.outputInternalGuideLayer, m_guide_layer.previousOutputInternalGuideLayer);
    }
    m_params.temporalModeUsePreviousLayers = 1;
}
}// namespace Pupil::optix