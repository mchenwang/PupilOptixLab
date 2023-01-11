#pragma once

#include "common/util.h"
#include "transform.h"

#include <unordered_map>
#include <string>
#include <memory>

#include <cuda_runtime.h>

namespace scene {
enum class ETextureAddressMode : unsigned int {
    Wrap = 0,
    Clamp = 1,
    Mirror = 2,
    Border = 3
};

enum ETextureFilterMode : unsigned int {
    Point = 0,
    Linear = 1
};

struct TextureDesc {
    ETextureAddressMode address_mode = ETextureAddressMode::Wrap;
    ETextureFilterMode filter_mode = ETextureFilterMode::Linear;
};

struct Texture {
    uint32_t w = 0;
    uint32_t h = 0;
    float *data = nullptr;
    cudaTextureObject_t cuda_obj = 0;
};

class TextureManager : public util::Singleton<TextureManager> {
private:
    struct ImageData {
        size_t w;
        size_t h;
        std::unique_ptr<float[]> data;

        ImageData(size_t w, size_t h) noexcept : w(w), h(h) {
            size_t data_size = w * h * 3;
            data = std::make_unique<float[]>(data_size);
        }
    };

    using MapDataType = std::pair<std::unique_ptr<ImageData>, cudaTextureObject_t>;
    std::unordered_map<std::string, MapDataType, util::StringHash, std::equal_to<>> m_image_datas;

    std::vector<cudaArray_t> m_cuda_memory_array;

    TextureManager() noexcept = default;

public:
    void LoadTextureFromFile(std::string_view) noexcept;
    [[nodiscard]] Texture GetColorTexture(float r, float g, float b) noexcept;
    [[nodiscard]] Texture GetTexture(std::string_view, TextureDesc desc = {}) noexcept;

    void Clear() noexcept;
};
}// namespace scene