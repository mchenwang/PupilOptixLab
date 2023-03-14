#include "texture.h"
#include "cuda_util/util.h"
#include "common/texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <iostream>

namespace scene {
// TODO: use dxtk for sRGB; exr
void TextureManager::LoadTextureFromFile(std::string_view file_path) noexcept {
    if (m_image_datas.find(file_path) != m_image_datas.end()) return;

    int isHdr = stbi_is_hdr(file_path.data());
    int w, h, c;

    std::unique_ptr<ImageData> image_data = nullptr;

    if (isHdr) {
        float *data = stbi_loadf(file_path.data(), &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * c;
            image_data = std::make_unique<ImageData>(w, h);
            for (size_t i = 0, j = 0; i < data_size; i += c) {
                image_data->data[j++] = data[i + 0];
                image_data->data[j++] = data[i + 1];
                image_data->data[j++] = data[i + 2];
                image_data->data[j++] = c == 4 ? data[i + 3] : 1.f;
            }
            stbi_image_free(data);
        }
    } else {
        unsigned char *data = stbi_load(file_path.data(), &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * c;
            image_data = std::make_unique<ImageData>(w, h);
            for (size_t i = 0, j = 0; i < data_size; i += c) {
                image_data->data[j++] = pow(data[i + 0] * 1.f / 255.f, 2.2f);
                image_data->data[j++] = pow(data[i + 1] * 1.f / 255.f, 2.2f);
                image_data->data[j++] = pow(data[i + 2] * 1.f / 255.f, 2.2f);
                image_data->data[j++] = c == 4 ? data[i + 3] * 1.f / 255.f : 1.f;
            }
            stbi_image_free(data);
        }
    }

    if (image_data == nullptr)
        std::cerr << "warring: fail to load image " << file_path << std::endl;
    else
        m_image_datas.emplace(file_path, std::move(image_data));
}

util::Texture TextureManager::GetColorTexture(float r, float g, float b) noexcept {
    util::Texture texture{};
    texture.type = util::ETextureType::RGB;
    texture.rgb.color.r = r;
    texture.rgb.color.g = g;
    texture.rgb.color.b = b;

    return texture;
}

util::Texture TextureManager::GetCheckerboardTexture(util::Float3 patch1, util::Float3 patch2) noexcept {
    util::Texture texture{};
    texture.type = util::ETextureType::Checkerboard;
    texture.checkerboard.patch1.r = patch1.r;
    texture.checkerboard.patch1.g = patch1.g;
    texture.checkerboard.patch1.b = patch1.b;
    texture.checkerboard.patch2.r = patch2.r;
    texture.checkerboard.patch2.g = patch2.g;
    texture.checkerboard.patch2.b = patch2.b;

    return texture;
}

util::Texture TextureManager::GetTexture(std::string_view id) noexcept {
    auto it = m_image_datas.find(id);
    if (it == m_image_datas.end()) {
        this->LoadTextureFromFile(id);
        it = m_image_datas.find(id);
        [[unlikely]] if (it == m_image_datas.end()) {
            return GetColorTexture(0.f, 0.f, 0.f);
        }
    }

    util::Texture texture{};
    texture.type = util::ETextureType::Bitmap;
    texture.bitmap.data = it->second->data.get();
    texture.bitmap.w = it->second->w;
    texture.bitmap.h = it->second->h;

    return texture;
}

void TextureManager::Clear() noexcept {
    m_image_datas.clear();
}
}// namespace scene