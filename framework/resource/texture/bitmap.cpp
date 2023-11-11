#include "../texture.h"
#include "render/texture.h"

#include "util/hash.h"
#include "util/id.h"

#include "cuda/stream.h"
#include "cuda/check.h"

namespace Pupil::resource {

    util::CountableRef<Texture> Bitmap::Make(std::string_view name) noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();
        return tex_mngr->Register(std::make_unique<Bitmap>(UserDisableTag{}, name));
    }

    util::CountableRef<Texture> Bitmap::Make(std::string_view path, bool sRGB, std::string_view name) noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();
        return tex_mngr->LoadTextureFromFile(path, sRGB, name);
    }

    Bitmap::Bitmap(UserDisableTag, std::string_view name) noexcept
        : Texture(name),
          m_width(0), m_height(0),
          m_address_mode(EAddressMode::Wrap), m_filter_mode(EFilterMode::Point),
          m_cuda_data_array(nullptr), m_cuda_tex_object(0) {
    }

    Bitmap::Bitmap(UserDisableTag,
                   std::string_view name,
                   size_t           w,
                   size_t           h,
                   const float*     data,
                   EAddressMode     address_mode,
                   EFilterMode      filter_mode) noexcept
        : Texture(name),
          m_width(w), m_height(h), m_address_mode(address_mode), m_filter_mode(filter_mode),
          m_cuda_data_array(nullptr), m_cuda_tex_object(0) {
        m_data = std::make_unique<float[]>(w * h * 4);
        SetData(data, w * h * 4 * sizeof(float), 0);
    }

    Bitmap::~Bitmap() noexcept {
        m_data.reset();
        CUDA_CHECK(cudaDestroyTextureObject(m_cuda_tex_object));
        CUDA_CHECK(cudaFreeArray(m_cuda_data_array));
        m_cuda_tex_object = 0;
        m_cuda_data_array = nullptr;
    }

    void* Bitmap::Clone() const noexcept {
        auto clone          = new Bitmap(UserDisableTag{}, m_name, m_width, m_height, m_data.get(), m_address_mode, m_filter_mode);
        clone->m_data_dirty = true;
        return clone;
    }

    uint64_t Bitmap::GetMemorySizeInByte() const noexcept {
        return m_width * m_height * sizeof(float) * 4;
    }

    void Bitmap::UploadToCuda() noexcept {
        if (m_cuda_tex_object && !m_data_dirty) return;

        size_t w = m_width;
        size_t h = m_height;

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaFreeArray(m_cuda_data_array));
        CUDA_CHECK(cudaMallocArray(&m_cuda_data_array, &channel_desc, w, h));

        size_t pitch  = w * 4 * sizeof(float);
        auto   stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::TextureUploading);
        CUDA_CHECK(cudaMemcpy2DToArrayAsync(m_cuda_data_array, 0, 0, m_data.get(), pitch, pitch, h, cudaMemcpyHostToDevice, *stream));

        cudaResourceDesc res_desc{};
        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = m_cuda_data_array;

        cudaTextureDesc cuda_texture_desc{};
        cuda_texture_desc.addressMode[0]      = (cudaTextureAddressMode)m_address_mode;
        cuda_texture_desc.addressMode[1]      = (cudaTextureAddressMode)m_address_mode;
        cuda_texture_desc.filterMode          = (cudaTextureFilterMode)m_filter_mode;
        cuda_texture_desc.readMode            = cudaReadModeElementType;
        cuda_texture_desc.normalizedCoords    = 1;
        cuda_texture_desc.maxAnisotropy       = 1;
        cuda_texture_desc.maxMipmapLevelClamp = 99;
        cuda_texture_desc.minMipmapLevelClamp = 0;
        cuda_texture_desc.mipmapFilterMode    = cudaFilterModePoint;
        cuda_texture_desc.borderColor[0]      = 1.0f;
        cuda_texture_desc.sRGB                = 0;

        CUDA_CHECK(cudaDestroyTextureObject(m_cuda_tex_object));
        CUDA_CHECK(cudaCreateTextureObject(&m_cuda_tex_object, &res_desc, &cuda_texture_desc, nullptr));
        m_data_dirty = false;
    }

    void Bitmap::SetAddressMode(EAddressMode address_mode) noexcept {
        if (m_address_mode == address_mode) return;

        m_address_mode = address_mode;
        m_data_dirty   = true;
    }

    void Bitmap::SetFilterMode(EFilterMode filter_mode) noexcept {
        if (m_filter_mode == filter_mode) return;

        m_filter_mode = filter_mode;
        m_data_dirty  = true;
    }

    void Bitmap::SetData(const float* data, size_t size_of_float, size_t offset_of_float) noexcept {
        if (data == nullptr) return;

        std::memcpy(m_data.get() + offset_of_float, data, size_of_float);
        m_data_dirty = true;
    }

    void Bitmap::SetSize(uint32_t w, uint32_t h) noexcept {
        if (w == m_width && h == m_height) return;

        m_width      = w;
        m_height     = h;
        m_data_dirty = true;
    }

    optix::Texture Bitmap::GetOptixTexture() noexcept {
        optix::Texture tex;
        tex.type   = optix::Texture::Bitmap;
        tex.bitmap = m_cuda_tex_object;
        return tex;
    }

    Float3 Bitmap::GetPixelAverage() const noexcept {
        float r = 0.f;
        float g = 0.f;
        float b = 0.f;
        for (int i = 0, idx = 0; i < m_height; ++i) {
            for (int j = 0; j < m_width; ++j) {
                r += m_data[idx++];
                g += m_data[idx++];
                b += m_data[idx++];
                idx++;// a
            }
        }
        return Float3(r, g, b) / (1.f * m_height * m_width);
    }
}// namespace Pupil::resource