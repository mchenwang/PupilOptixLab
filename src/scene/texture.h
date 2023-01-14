#pragma once

#include "common/util.h"

#include <unordered_map>
#include <string>
#include <memory>

#include <cuda_runtime.h>

namespace util {
struct Texture;
}

namespace scene {
class TextureManager : public util::Singleton<TextureManager> {
private:
    struct ImageData {
        size_t w;
        size_t h;
        std::unique_ptr<float[]> data;

        ImageData(size_t w, size_t h) noexcept : w(w), h(h) {
            size_t data_size = w * h * 4;
            data = std::make_unique<float[]>(data_size);
        }
    };

    std::unordered_map<std::string, std::unique_ptr<ImageData>, util::StringHash, std::equal_to<>> m_image_datas;

public:
    TextureManager() noexcept = default;

    void LoadTextureFromFile(std::string_view) noexcept;
    [[nodiscard]] util::Texture GetColorTexture(float r, float g, float b) noexcept;
    [[nodiscard]] util::Texture GetCheckerboardTexture(float patch1[3], float patch2[3]) noexcept;
    [[nodiscard]] util::Texture GetTexture(std::string_view) noexcept;

    void Clear() noexcept;
};
}// namespace scene