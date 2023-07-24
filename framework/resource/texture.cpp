#include "texture.h"
#include "cuda/util.h"
#include "util/texture.h"
#include "util/log.h"

#include <iostream>

namespace Pupil::resource {
void TextureManager::LoadTextureFromFile(std::string_view file_path) noexcept {
    if (m_image_datas.find(file_path) != m_image_datas.end()) return;
    auto image = util::BitmapTexture::Load(file_path);
    if (image.data != nullptr) {
        std::unique_ptr<ImageData> image_data =
            std::make_unique<ImageData>(image.w, image.h, image.data);

        m_image_datas.emplace(file_path, std::move(image_data));
    }
}

util::Texture TextureManager::GetColorTexture(float r, float g, float b) noexcept {
    util::Texture texture{};
    texture.type = util::ETextureType::RGB;
    texture.rgb.color.r = r;
    texture.rgb.color.g = g;
    texture.rgb.color.b = b;

    return texture;
}

util::Texture TextureManager::GetColorTexture(util::Float3 color) noexcept {
    util::Texture texture{};
    texture.type = util::ETextureType::RGB;
    texture.rgb.color = color;

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
}// namespace Pupil::resource