#include "scene/emitter.h"

#include "cuda/check.h"
#include "cuda/stream.h"
#include "cuda/util.h"

namespace Pupil {
    EnvmapEmitter::EnvmapEmitter(const resource::TextureInstance& radiance) noexcept
        : Emitter(radiance, Transform{}), m_scale(1.f), m_cuda_memory(0) {}

    EnvmapEmitter::~EnvmapEmitter() noexcept {
        CUDA_FREE(m_cuda_memory);
    }

    // void EnvmapEmitter::UploadToCuda() noexcept {
    //     if (m_cuda_memory) return;

    //     m_radiance->UploadToCuda();

    //     auto [w, h]                           = m_radiance.GetTexture().As<resource::Bitmap>()->GetSize();
    //     std::unique_ptr<float[]> m_col_cdf    = std::make_unique<float[]>((w + 1) * h);
    //     std::unique_ptr<float[]> m_row_cdf    = std::make_unique<float[]>(h + 1);
    //     std::unique_ptr<float[]> m_row_weight = std::make_unique<float[]>(h);

    //     auto data = m_radiance.GetTexture().As<resource::Bitmap>()->GetData();

    //     size_t col_index = 0, row_index = 0;
    //     float  row_sum         = 0.f;
    //     m_row_cdf[row_index++] = 0.f;
    //     for (auto y = 0u; y < h; ++y) {
    //         float col_sum          = 0.f;
    //         m_col_cdf[col_index++] = 0.f;
    //         for (auto x = 0u; x < w; ++x) {
    //             auto pixel_index = y * w + x;
    //             auto r           = data[pixel_index * 4 + 0];
    //             auto g           = data[pixel_index * 4 + 1];
    //             auto b           = data[pixel_index * 4 + 2];
    //             col_sum += optix::GetLuminance(make_float3(r, g, b));
    //             m_col_cdf[col_index++] = col_sum;
    //         }

    //         for (auto x = 1u; x < w; ++x)
    //             m_col_cdf[col_index - x - 1] /= col_sum;
    //         m_col_cdf[col_index - 1] = 1.f;

    //         float weight    = std::sin((y + 0.5f) * M_PIf / h);
    //         m_row_weight[y] = weight;
    //         row_sum += col_sum * weight;
    //         m_row_cdf[row_index++] = row_sum;
    //     }

    //     for (auto y = 1u; y < h; ++y)
    //         m_row_cdf[row_index - y - 1] /= row_sum;
    //     m_row_cdf[row_index - 1] = 1.f;

    //     m_normalization = 1.f / (row_sum * (2.f * M_PIf / w) * (M_PIf / h));
    //     m_width         = w;
    //     m_height        = h;

    //     auto size   = sizeof(float) * ((w + 1) * h + h + 1 + h);
    //     auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::EmitterUploading);

    //     CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_cuda_memory), size, *stream));

    //     auto cuda_col_cdf    = m_cuda_memory;
    //     auto cuda_row_cdf    = cuda_col_cdf + sizeof(float) * ((w + 1) * h);
    //     auto cuda_row_weight = cuda_row_cdf + sizeof(float) * (h + 1);

    //     CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_col_cdf), m_col_cdf.get(), sizeof(float) * ((w + 1) * h), cudaMemcpyHostToDevice, *stream));
    //     CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_row_cdf), m_row_cdf.get(), sizeof(float) * (h + 1), cudaMemcpyHostToDevice, *stream));
    //     CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_row_weight), m_row_weight.get(), sizeof(float) * h, cudaMemcpyHostToDevice, *stream));
    // }

    void EnvmapEmitter::UploadToCuda() noexcept {
        if (m_cuda_memory) return;

        m_radiance->UploadToCuda();

        auto [w, h] = m_radiance.GetTexture().As<resource::Bitmap>()->GetSize();
        std::vector<float> m_col_cdf((w + 1) * h);
        std::vector<float> m_row_cdf(h + 1);
        std::vector<float> m_row_weight(h);

        auto data = m_radiance.GetTexture().As<resource::Bitmap>()->GetData();

        size_t col_index = 0, row_index = 0;
        float  row_sum         = 0.f;
        m_row_cdf[row_index++] = 0.f;
        for (auto y = 0u; y < h; ++y) {
            float col_sum          = 0.f;
            m_col_cdf[col_index++] = 0.f;
            for (auto x = 0u; x < w; ++x) {
                auto pixel_index = y * w + x;
                auto r           = data[pixel_index * 4 + 0];
                auto g           = data[pixel_index * 4 + 1];
                auto b           = data[pixel_index * 4 + 2];
                col_sum += optix::GetLuminance(make_float3(r, g, b));
                m_col_cdf[col_index++] = col_sum;
            }

            for (auto x = 1u; x < w; ++x)
                m_col_cdf[col_index - x - 1] /= col_sum;
            m_col_cdf[col_index - 1] = 1.f;

            float weight    = std::sin((y + 0.5f) * M_PIf / h);
            m_row_weight[y] = weight;
            row_sum += col_sum * weight;
            m_row_cdf[row_index++] = row_sum;
        }

        for (auto y = 1u; y < h; ++y)
            m_row_cdf[row_index - y - 1] /= row_sum;
        m_row_cdf[row_index - 1] = 1.f;

        m_normalization = 1.f / (row_sum * (2.f * M_PIf / w) * (M_PIf / h));
        m_width         = w;
        m_height        = h;

        auto size   = sizeof(float) * ((w + 1) * h + h + 1 + h);
        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::EmitterUploading);

        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_cuda_memory), size, *stream));

        auto cuda_col_cdf    = m_cuda_memory;
        auto cuda_row_cdf    = cuda_col_cdf + sizeof(float) * ((w + 1) * h);
        auto cuda_row_weight = cuda_row_cdf + sizeof(float) * (h + 1);

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_col_cdf), m_col_cdf.data(), sizeof(float) * ((w + 1) * h), cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_row_cdf), m_row_cdf.data(), sizeof(float) * (h + 1), cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cuda_row_weight), m_row_weight.data(), sizeof(float) * h, cudaMemcpyHostToDevice, *stream));
    }

    optix::Emitter EnvmapEmitter::GetOptixEmitter() noexcept {
        UploadToCuda();
        optix::Emitter emitter;
        emitter.type                  = optix::EEmitterType::EnvMap;
        emitter.env_map.scale         = m_scale;
        emitter.env_map.normalization = m_normalization;
        emitter.env_map.map_size      = make_uint2(m_width, m_height);
        emitter.env_map.radiance      = m_radiance.GetOptixTexture();

        auto cuda_col_cdf    = m_cuda_memory;
        auto cuda_row_cdf    = cuda_col_cdf + sizeof(float) * ((m_width + 1) * m_height);
        auto cuda_row_weight = cuda_row_cdf + sizeof(float) * (m_height + 1);

        emitter.env_map.col_cdf.SetData(cuda_col_cdf, (m_width + 1) * m_height);
        emitter.env_map.row_cdf.SetData(cuda_row_cdf, m_height + 1);
        emitter.env_map.row_weight.SetData(cuda_row_weight, m_height);

        auto to_world = m_transform.GetMatrix4x4();

        emitter.env_map.to_world.r0 = Pupil::cuda::MakeFloat3(to_world.r0);
        emitter.env_map.to_world.r1 = Pupil::cuda::MakeFloat3(to_world.r1);
        emitter.env_map.to_world.r2 = Pupil::cuda::MakeFloat3(to_world.r2);

        auto to_local               = Pupil::Inverse(to_world);
        emitter.env_map.to_local.r0 = Pupil::cuda::MakeFloat3(to_local.r0);
        emitter.env_map.to_local.r1 = Pupil::cuda::MakeFloat3(to_local.r1);
        emitter.env_map.to_local.r2 = Pupil::cuda::MakeFloat3(to_local.r2);
        return emitter;
    }

    ConstEmitter::ConstEmitter(const Float3& radiance) noexcept {
        m_radiance  = resource::RGBTexture::Make(radiance, "const environment");
        m_transform = Transform{};
    }

    ConstEmitter::~ConstEmitter() noexcept {
    }

    optix::Emitter ConstEmitter::GetOptixEmitter() noexcept {
        optix::Emitter emitter;
        emitter.type            = optix::EEmitterType::ConstEnv;
        auto color              = m_radiance.GetTexture().As<resource::RGBTexture>()->GetColor();
        emitter.const_env.color = make_float3(color.x, color.y, color.z);
        return emitter;
    }
}// namespace Pupil