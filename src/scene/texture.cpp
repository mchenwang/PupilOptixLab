#include "texture.h"
#include "cuda_util/util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <iostream>
#include <algorithm>

namespace scene {
void TextureManager::LoadTextureFromFile(std::string_view file_path) noexcept {
    if (m_image_datas.find(file_path) != m_image_datas.end()) return;

    int isHdr = stbi_is_hdr(file_path.data());
    int w, h, c;

    std::unique_ptr<ImageData> image_data = nullptr;

    if (isHdr) {
        float *data = stbi_loadf(file_path.data(), &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * 3;
            image_data = std::make_unique<ImageData>(w, h);
            std::transform(data, data + data_size, image_data->data.get(), [](float c) { return c; });
            stbi_image_free(data);
        }
    } else {
        unsigned char *data = stbi_load(file_path.data(), &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * 3;
            image_data = std::make_unique<ImageData>(w, h);
            std::transform(data, data + data_size, image_data->data.get(), [](unsigned char c) { return c; });
            stbi_image_free(data);
        }
    }

    if (image_data == nullptr) 
        std::cerr << "warring: fail to load image " << file_path << std::endl;
    else 
        m_image_datas.emplace(file_path, std::make_pair(std::move(image_data), 0));
}

Texture TextureManager::GetColorTexture(float r, float g, float b) noexcept {

}

Texture TextureManager::GetTexture(std::string_view id, TextureDesc texture_desc) noexcept {
    decltype(m_image_datas)::iterator image_data_it = m_image_datas.find(id);
    if (image_data_it == m_image_datas.end()) {
        return GetColorTexture(0.f, 0.f, 0.f);
    }

    auto image_data = image_data_it->second.first.get();

    Texture texture{
        .w = image_data->w,
        .h = image_data->h,
        .data = image_data->data.get(),
        .cuda_obj = image_data_it->second.second
    };

    if (texture.cuda_obj == 0) {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaArray_t cuda_array;
        cudaMallocArray(&cuda_array, &channel_desc, texture.w, texture.h);

        size_t size = texture.w * texture.h * 3 * sizeof(float);
        cudaMemcpyToArray(cuda_array, 0, 0, texture.data, size, cudaMemcpyHostToDevice);

        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;

        cudaTextureDesc cuda_texture_desc{};
        cuda_texture_desc.addressMode[0] = (cudaTextureAddressMode)texture_desc.address_mode;
        cuda_texture_desc.addressMode[1] = (cudaTextureAddressMode)texture_desc.address_mode;
        cuda_texture_desc.filterMode = (cudaTextureFilterMode)texture_desc.filter_mode;
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

        image_data_it->second.second = cuda_tex;
        texture.cuda_obj = cuda_tex;
        m_cuda_memory_array.push_back(cuda_array);
    }

    return texture;
}

void TextureManager::Clear() noexcept {
    for (cudaArray_t &data : m_cuda_memory_array)
        CUDA_CHECK(cudaFreeArray(data));
    m_cuda_memory_array.clear();
    m_image_datas.clear();
}
}// namespace scene