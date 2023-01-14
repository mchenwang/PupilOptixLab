#include "cuda_texture.h"
#include "cuda_util/util.h"
#include "common/texture.h"

#include <unordered_map>

namespace {
struct CudaTextureHash {
    [[nodiscard]] size_t operator()(const util::Texture &texture) const {
        // http://stackoverflow.com/a/1646913/126995
        size_t res = 17;
        res = res * 31 + std::hash<int>()((int)texture.type);
        switch (texture.type) {
            case util::ETextureType::RGB:
                res = res * 31 + std::hash<float>()(texture.rgb.color.r);
                res = res * 31 + std::hash<float>()(texture.rgb.color.g);
                res = res * 31 + std::hash<float>()(texture.rgb.color.b);
                break;
            case util::ETextureType::Checkerboard:
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch1.r);
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch1.g);
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch1.b);
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch2.r);
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch2.g);
                res = res * 31 + std::hash<float>()(texture.checkerboard.patch2.b);
                break;
            case util::ETextureType::Bitmap:
                res = res * 31 + std::hash<float *>()(texture.bitmap.data);
                res = res * 31 + std::hash<size_t>()(texture.bitmap.w);
                res = res * 31 + std::hash<size_t>()(texture.bitmap.h);
                res = res * 31 + std::hash<int>()((int)texture.bitmap.address_mode);
                res = res * 31 + std::hash<int>()((int)texture.bitmap.filter_mode);
                break;
        }
        return res;
    }
};

struct TextureCmp {
    [[nodiscard]] constexpr bool operator()(const util::Texture &a, const util::Texture &b) const {
        if (a.type != b.type) return false;
        switch (a.type) {
            case util::ETextureType::RGB:
                return a.rgb.color.r == b.rgb.color.r && a.rgb.color.g == b.rgb.color.g && a.rgb.color.b == b.rgb.color.b;
            case util::ETextureType::Checkerboard:
                return a.checkerboard.patch1.r == b.checkerboard.patch1.r && a.checkerboard.patch1.g == b.checkerboard.patch1.g && a.checkerboard.patch1.b == b.checkerboard.patch1.b &&
                       a.checkerboard.patch2.r == b.checkerboard.patch2.r && a.checkerboard.patch2.g == b.checkerboard.patch2.g && a.checkerboard.patch2.b == b.checkerboard.patch2.b;
            case util::ETextureType::Bitmap:
                return a.bitmap.data == b.bitmap.data && a.bitmap.w == b.bitmap.w && a.bitmap.h == b.bitmap.h;
        }
        return false;
    }
};

std::unordered_map<util::Texture, cudaTextureObject_t, CudaTextureHash, TextureCmp> m_cuda_texture_map;
}// namespace

namespace device {
cudaTextureObject_t CudaTextureManager::GetCudaTexture(util::Texture texture) noexcept {
    if (texture.type != util::ETextureType::Bitmap) return 0;
    
    auto it = m_cuda_texture_map.find(texture);
    if (it != m_cuda_texture_map.end()) return it->second;

    size_t w = 1;
    size_t h = 1;

    w = texture.bitmap.w;
    h = texture.bitmap.h;

    //if (texture.type == util::ETextureType::Bitmap) {
    //    w = texture.bitmap.w;
    //    h = texture.bitmap.h;
    //} else if (texture.type == util::ETextureType::Checkerboard) {
    //    w = 2;
    //    h = 2;
    //}

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuda_array;
    cudaMallocArray(&cuda_array, &channel_desc, w, h);

    size_t size = w * h * 4 * sizeof(float);
    size_t pitch = w * 4 * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, texture.bitmap.data, pitch, pitch, h, cudaMemcpyHostToDevice));

    //if (texture.type == util::ETextureType::Bitmap) {
    //    size_t size = w * h * 4 * sizeof(float);
    //    size_t pitch = w * 4 * sizeof(float);
    //    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, texture.bitmap.data, pitch, pitch, h, cudaMemcpyHostToDevice));
    //} else if (texture.type == util::ETextureType::RGB) {
    //    float data[4]{ texture.rgb.color.r, texture.rgb.color.g, texture.rgb.color.b, 255.f };
    //    size_t size = w * h * 4 * sizeof(float);
    //    size_t pitch = w * 4 * sizeof(float);
    //    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, data, pitch, pitch, h, cudaMemcpyHostToDevice));
    //} else {
    //    auto [r1, g1, b1] = texture.checkerboard.patch1;
    //    auto [r2, g2, b2] = texture.checkerboard.patch2;
    //    float data[]{
    //        r1, g1, b1, 255.f, r2, g2, b2, 255.f,
    //        r2, g2, b2, 255.f, r1, g1, b1, 255.f
    //    };
    //    size_t size = w * h * 4 * sizeof(float);
    //    size_t pitch = w * 4 * sizeof(float);
    //    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, data, pitch, pitch, h, cudaMemcpyHostToDevice));
    //}

    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc cuda_texture_desc{};
    cuda_texture_desc.addressMode[0] = (cudaTextureAddressMode)texture.bitmap.address_mode;
    cuda_texture_desc.addressMode[1] = (cudaTextureAddressMode)texture.bitmap.address_mode;
    cuda_texture_desc.filterMode = (cudaTextureFilterMode)texture.bitmap.filter_mode;
    cuda_texture_desc.readMode = cudaReadModeNormalizedFloat;
    cuda_texture_desc.normalizedCoords = 1;
    cuda_texture_desc.maxAnisotropy = 1;
    cuda_texture_desc.maxMipmapLevelClamp = 99;
    cuda_texture_desc.minMipmapLevelClamp = 0;
    cuda_texture_desc.mipmapFilterMode = cudaFilterModePoint;
    cuda_texture_desc.borderColor[0] = 1.0f;
    cuda_texture_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &cuda_texture_desc, nullptr));
    it->second = cuda_tex;

    m_cuda_memory_array.push_back(cuda_array);

    return cuda_tex;
}

void CudaTextureManager::Clear() noexcept {
    for (auto &&[key, value] : m_cuda_texture_map) {
        CUDA_CHECK(cudaDestroyTextureObject(value));
    }
    m_cuda_texture_map.clear();

    for (auto &&data : m_cuda_memory_array) {
        CUDA_CHECK(cudaFreeArray(data));
    }
    m_cuda_memory_array.clear();
}
}// namespace device