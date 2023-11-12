#include "denoiser.h"
#include "context.h"
#include "check.h"
#include "cuda/check.h"
#include "util/log.h"

#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

namespace {
    inline bool FillOptixImageStruct(OptixImage2D& ret, CUdeviceptr data, unsigned int w, unsigned h, bool upscale = false) {
        if (!data) [[unlikely]]
            return false;

        ret = {
            .data               = data,
            .width              = w * (upscale ? 2 : 1),
            .height             = h * (upscale ? 2 : 1),
            .rowStrideInBytes   = static_cast<unsigned int>(w * (upscale ? 2 : 1) * sizeof(float4)),
            .pixelStrideInBytes = sizeof(float4),
            .format             = OPTIX_PIXEL_FORMAT_FLOAT4};
        return true;
    }
}// namespace

namespace Pupil::optix {
    struct Denoiser::Impl {
        util::CountableRef<cuda::Stream> stream;
        OptixDenoiser                    denoiser = nullptr;
        OptixDenoiserParams              params{};
        OptixDenoiserGuideLayer          guide_layer{};

        CUdeviceptr  scratch      = 0;
        size_t       scratch_size = 0;
        unsigned int overlap      = 0;

        CUdeviceptr state      = 0;
        size_t      state_size = 0;

        CUdeviceptr hdr_intensity     = 0;
        CUdeviceptr hdr_average_color = 0;

        unsigned int mode = EMode::UseAlbedo | EMode::UseNormal;

        unsigned int input_w = 0;
        unsigned int input_h = 0;
        unsigned int tile_w  = 100;
        unsigned int tile_h  = 100;
    };

    Denoiser::Denoiser(util::CountableRef<cuda::Stream> stream, unsigned int mode) noexcept {
        m_impl         = new Impl();
        m_impl->stream = stream;
        SetMode(mode);
    }
    Denoiser::~Denoiser() noexcept { Destroy(); }

    Denoiser::operator OptixDenoiser() const noexcept { return m_impl->denoiser; }

    void Denoiser::Destroy() noexcept {
        if (m_impl->denoiser)
            OPTIX_CHECK(optixDenoiserDestroy(m_impl->denoiser));
        m_impl->denoiser = nullptr;

        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::None);

        CUDA_FREE_ASYNC(m_impl->hdr_intensity, *stream);
        CUDA_FREE_ASYNC(m_impl->hdr_average_color, *stream);
        CUDA_FREE_ASYNC(m_impl->guide_layer.outputInternalGuideLayer.data, *stream);
        CUDA_FREE_ASYNC(m_impl->guide_layer.previousOutputInternalGuideLayer.data, *stream);
        CUDA_FREE_ASYNC(m_impl->scratch, *stream);
        CUDA_FREE_ASYNC(m_impl->state, *stream);
    }

    void Denoiser::SetMode(unsigned int mode) noexcept {
        if (m_impl->denoiser) {
            if (mode == m_impl->mode)
                return;

            OPTIX_CHECK(optixDenoiserDestroy(m_impl->denoiser));
            m_impl->denoiser = nullptr;
        }

        m_impl->mode = mode;

        OptixDenoiserOptions options{
            .guideAlbedo = mode & EMode::UseAlbedo,
            .guideNormal = mode & EMode::UseNormal};

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
        OPTIX_CHECK(optixDenoiserCreate(*ctx, kind, &options, &m_impl->denoiser));
    }

    void Denoiser::SetTile(unsigned int tile_w, unsigned int tile_h) noexcept {
        m_impl->tile_w = tile_w;
        m_impl->tile_h = tile_h;
    }

    void Denoiser::Setup(unsigned int w, unsigned int h) noexcept {
        if (w == 0 || h == 0) {
            Log::Error("Optix Denoiser: size must be >0.");
            return;
        }
        m_impl->input_w = w;
        m_impl->input_h = h;

        if (m_impl->tile_w > m_impl->input_w || m_impl->tile_h > m_impl->input_h) {
            Log::Warn("Optix Denoiser: tile size({}x{}) must be smaller than input film size({}x{}).",
                      m_impl->tile_w, m_impl->tile_h, m_impl->input_w, m_impl->input_h);

            m_impl->tile_w = w;
            m_impl->tile_h = h;
        }

        auto size_w = m_impl->mode & EMode::Tiled ? m_impl->tile_w : w;
        auto size_h = m_impl->mode & EMode::Tiled ? m_impl->tile_h : h;

        OptixDenoiserSizes sizes{};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_impl->denoiser, size_w, size_h, &sizes));

        if (m_impl->mode & EMode::Tiled) {
            m_impl->scratch_size = sizes.withOverlapScratchSizeInBytes;
            m_impl->overlap      = sizes.overlapWindowSizeInPixels;
        } else {
            m_impl->scratch_size = sizes.withoutOverlapScratchSizeInBytes;
            m_impl->overlap      = 0;
        }

        if (m_impl->mode & EMode::UseUpscale2X || m_impl->mode & EMode::ApplyToAOV) {
            if (m_impl->hdr_average_color == 0)
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->hdr_average_color), 3 * sizeof(float), *m_impl->stream));
        } else {
            if (m_impl->hdr_intensity == 0)
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->hdr_intensity), sizeof(float), *m_impl->stream));
        }

        CUDA_FREE_ASYNC(m_impl->scratch, *m_impl->stream);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->scratch), m_impl->scratch_size, *m_impl->stream));

        m_impl->state_size = sizes.stateSizeInBytes;
        CUDA_FREE_ASYNC(m_impl->state, *m_impl->stream);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_impl->state), m_impl->state_size, *m_impl->stream));

        OPTIX_CHECK(optixDenoiserSetup(
            m_impl->denoiser,
            *m_impl->stream,
            m_impl->mode & EMode::Tiled ? m_impl->tile_w + 2 * m_impl->overlap : w,
            m_impl->mode & EMode::Tiled ? m_impl->tile_h + 2 * m_impl->overlap : h,
            m_impl->state,
            m_impl->state_size,
            m_impl->scratch,
            m_impl->scratch_size));

        m_impl->params.denoiseAlpha                  = OPTIX_DENOISER_ALPHA_MODE_COPY;
        m_impl->params.hdrIntensity                  = m_impl->hdr_intensity;
        m_impl->params.hdrAverageColor               = m_impl->hdr_average_color;
        m_impl->params.blendFactor                   = 0.0f;
        m_impl->params.temporalModeUsePreviousLayers = 0;

        if (m_impl->mode & EMode::UseTemporal) {
            CUdeviceptr internal_memory_in  = 0;
            CUdeviceptr internal_memory_out = 0;
            size_t      internal_size       = (m_impl->mode & EMode::Tiled ? 4 : 1) *
                                   m_impl->input_h * m_impl->input_w *
                                   sizes.internalGuideLayerPixelSizeInBytes;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&internal_memory_in), internal_size, *m_impl->stream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&internal_memory_out), internal_size, *m_impl->stream));
            CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(internal_memory_in), 0, internal_size, *m_impl->stream));

            CUDA_FREE_ASYNC(m_impl->guide_layer.previousOutputInternalGuideLayer.data, *m_impl->stream);
            CUDA_FREE_ASYNC(m_impl->guide_layer.outputInternalGuideLayer.data, *m_impl->stream);

            m_impl->guide_layer.previousOutputInternalGuideLayer.data               = internal_memory_in;
            m_impl->guide_layer.previousOutputInternalGuideLayer.width              = (m_impl->mode & EMode::Tiled ? 2 : 1) * m_impl->input_w;
            m_impl->guide_layer.previousOutputInternalGuideLayer.height             = (m_impl->mode & EMode::Tiled ? 2 : 1) * m_impl->input_h;
            m_impl->guide_layer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned(sizes.internalGuideLayerPixelSizeInBytes);
            m_impl->guide_layer.previousOutputInternalGuideLayer.rowStrideInBytes   = m_impl->guide_layer.previousOutputInternalGuideLayer.width *
                                                                                    m_impl->guide_layer.previousOutputInternalGuideLayer.pixelStrideInBytes;
            m_impl->guide_layer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

            m_impl->guide_layer.outputInternalGuideLayer      = m_impl->guide_layer.previousOutputInternalGuideLayer;
            m_impl->guide_layer.outputInternalGuideLayer.data = internal_memory_out;
        }
    }

    void Denoiser::Execute(const ExecutionData& data) noexcept {
        if (m_impl->input_h == 0 || m_impl->input_w == 0) [[unlikely]] {
            Log::Warn("Optix Denoiser: does not setup.");
            return;
        }

        OptixDenoiserLayer layer{};
        if (!FillOptixImageStruct(layer.input, data.input, m_impl->input_w, m_impl->input_h)) {
            Log::Error("Optix Denoiser: input ptr can not be NULL.");
            return;
        }

        if (!FillOptixImageStruct(layer.output, data.output, m_impl->input_w, m_impl->input_h, m_impl->mode & EMode::UseUpscale2X)) {
            Log::Error("Optix Denoiser: output ptr can not be NULL.");
            return;
        }

        if (m_impl->mode & EMode::UseAlbedo) {
            if (!FillOptixImageStruct(m_impl->guide_layer.albedo, data.albedo, m_impl->input_w, m_impl->input_h)) {
                Log::Error("Optix Denoiser: input albedo ptr can not be NULL.");
                return;
            }
        }

        if (m_impl->mode & EMode::UseNormal) {
            if (!FillOptixImageStruct(m_impl->guide_layer.normal, data.normal, m_impl->input_w, m_impl->input_h)) {
                Log::Error("Optix Denoiser: input normal ptr can not be NULL.");
                return;
            }
        }

        if (m_impl->mode & EMode::UseTemporal) {
            if (!FillOptixImageStruct(layer.previousOutput, data.prev_output, m_impl->input_w, m_impl->input_h, m_impl->mode & EMode::UseUpscale2X)) {
                Log::Error("Optix Denoiser: input prev_output ptr can not be NULL.");
                return;
            }
            if (!FillOptixImageStruct(m_impl->guide_layer.flow, data.motion_vector, m_impl->input_w, m_impl->input_h, m_impl->mode & EMode::UseUpscale2X)) {
                Log::Error("Optix Denoiser: input motion_vector ptr can not be NULL.");
                return;
            }
        }

        if (m_impl->hdr_intensity) {
            OPTIX_CHECK(optixDenoiserComputeIntensity(
                m_impl->denoiser,
                *m_impl->stream,
                &layer.input,
                m_impl->hdr_intensity,
                m_impl->scratch,
                m_impl->scratch_size));
        }
        if (m_impl->hdr_average_color) {
            OPTIX_CHECK(optixDenoiserComputeAverageColor(
                m_impl->denoiser,
                *m_impl->stream,
                &layer.input,
                m_impl->hdr_average_color,
                m_impl->scratch,
                m_impl->scratch_size));
        }

        if (m_impl->mode & EMode::Tiled) {
            OPTIX_CHECK(optixUtilDenoiserInvokeTiled(
                m_impl->denoiser,
                *m_impl->stream,
                &m_impl->params,
                m_impl->state,
                m_impl->state_size,
                &m_impl->guide_layer,
                &layer,
                1,
                m_impl->scratch,
                m_impl->scratch_size,
                m_impl->overlap,
                m_impl->tile_w,
                m_impl->tile_h));
        } else {
            OPTIX_CHECK(optixDenoiserInvoke(
                m_impl->denoiser,
                *m_impl->stream,
                &m_impl->params,
                m_impl->state,
                m_impl->state_size,
                &m_impl->guide_layer,
                &layer,
                1,
                0,// input offset X
                0,// input offset y
                m_impl->scratch,
                m_impl->scratch_size));
        }

        if (m_impl->mode & EMode::UseTemporal) {
            std::swap(m_impl->guide_layer.outputInternalGuideLayer, m_impl->guide_layer.previousOutputInternalGuideLayer);
        }
        m_impl->params.temporalModeUsePreviousLayers = 1;
    }
}// namespace Pupil::optix