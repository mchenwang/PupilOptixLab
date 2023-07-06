#include "texture.h"
#include "log.h"

#include <filesystem>

#include "tinyexr.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace {
bool SaveHdr(float *data, size_t w, size_t h, const char *save_path) noexcept {
    stbi_flip_vertically_on_write(true);
    int ret = stbi_write_hdr(save_path, w, h, 4, data);
    if (ret == 1) {
        Pupil::Log::Info("image was saved successfully in [{}].", save_path);
    } else {
        Pupil::Log::Warn("image saving failed.");
    }
    return ret == 1;
}
bool SaveExr(float *data, size_t w, size_t h, const char *save_path) noexcept {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    exr_image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(w * h);
    images[1].resize(w * h);
    images[2].resize(w * h);

    for (int y = h - 1, wi = 0; y >= 0; --y) {
        for (int x = 0; x < w; ++x, ++wi) {
            int i = y * w + x;
            images[0][wi] = data[4 * i + 0];
            images[1][wi] = data[4 * i + 1];
            images[2][wi] = data[4 * i + 2];
        }
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0));// B
    image_ptr[1] = &(images[1].at(0));// G
    image_ptr[2] = &(images[0].at(0));// R

    exr_image.images = (unsigned char **)image_ptr;
    exr_image.width = w;
    exr_image.height = h;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be BGR(A) order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    const char *err;
    int ret = SaveEXRImageToFile(&exr_image, &header, save_path, &err);
    if (ret == TINYEXR_SUCCESS) {
        Pupil::Log::Info("image was saved successfully in [{}].", save_path);
    } else {
        Pupil::Log::Warn("image saving failed({}).", err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    return ret == TINYEXR_SUCCESS;
}

Pupil::util::BitmapTexture StbImageLoad(const char *path) noexcept {
    int is_hdr = stbi_is_hdr(path);
    int w, h, c;

    Pupil::util::BitmapTexture image;

    if (is_hdr) {
        float *data = stbi_loadf(path, &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * c;
            image.data = new float[w * h * 4];
            for (size_t i = 0, j = 0; i < data_size; i += c) {
                image.data[j++] = data[i + 0];
                image.data[j++] = data[i + 1];
                image.data[j++] = data[i + 2];
                image.data[j++] = c == 4 ? data[i + 3] : 1.f;
            }
            stbi_image_free(data);
        }
    } else {
        unsigned char *data = stbi_load(path, &w, &h, &c, 0);
        if (data) {
            size_t data_size = static_cast<size_t>(w) * h * c;
            image.data = new float[w * h * 4];
            for (size_t i = 0, j = 0; i < data_size; i += c) {
                image.data[j++] = pow(data[i + 0] * 1.f / 255.f, 2.2f);
                image.data[j++] = pow(data[i + 1] * 1.f / 255.f, 2.2f);
                image.data[j++] = pow(data[i + 2] * 1.f / 255.f, 2.2f);
                image.data[j++] = c == 4 ? data[i + 3] * 1.f / 255.f : 1.f;
            }
            stbi_image_free(data);
        }
    }

    image.w = static_cast<size_t>(w);
    image.h = static_cast<size_t>(h);

    if (image.data == nullptr)
        Pupil::Log::Warn("fail to load image [{}]", path);
    else
        Pupil::Log::Info("load image [{}] ({}x{})", path, image.w, image.h);
    return image;
}

Pupil::util::BitmapTexture ExrImageLoad(const char *path) noexcept {
    float *data;
    const char *err;
    int w, h;
    if (LoadEXR(&data, &w, &h, path, &err) != 0) {
        Pupil::Log::Warn("fail to load exr image [{}] with error: [{}].", path, err);
        return {};
    }
    Pupil::util::BitmapTexture image;
    image.w = static_cast<size_t>(w);
    image.h = static_cast<size_t>(h);
    image.data = new float[w * h * 4];
    std::memcpy(image.data, data, sizeof(float) * w * h * 4);
    free(data);

    Pupil::Log::Info("load image [{}] ({}x{})", path, image.w, image.h);
    return image;
}
}// namespace

namespace Pupil::util {
bool BitmapTexture::Save(float *data, size_t w, size_t h, std::string_view save_path, FileFormat format) noexcept {
    switch (format) {
        case FileFormat::HDR:
            return SaveHdr(data, w, h, save_path.data());
        case FileFormat::EXR:
            return SaveExr(data, w, h, save_path.data());
    }
    Pupil::Log::Warn("Unknown bitmap format(BitmapTexture::Save).");
    return false;
}

BitmapTexture BitmapTexture::Load(std::string_view file_path) noexcept {
    BitmapTexture ret;
    std::filesystem::path path{ file_path };

    static std::filesystem::path exr{ ".exr" };

    if (path.extension() == exr) {
        return ExrImageLoad(file_path.data());
    } else {
        return StbImageLoad(file_path.data());
    }
}
}// namespace Pupil::util