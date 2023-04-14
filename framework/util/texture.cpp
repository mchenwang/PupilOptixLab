#include "texture.h"
#include "log.h"

#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace {
bool SaveHdr(float *data, size_t w, size_t h, const char *save_path) noexcept {
    stbi_flip_vertically_on_write(true);
    int ret = stbi_write_hdr(save_path, w, h, 4, data);
    if (ret == 1) {
        Pupil::Log::Info("image was saved successfully in [{}].\n", save_path);
    } else {
        Pupil::Log::Warn("image saving failed.\n");
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
        Pupil::Log::Info("image was saved successfully in [{}].\n", save_path);
    } else {
        Pupil::Log::Warn("image saving failed({}).\n", err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    return ret == TINYEXR_SUCCESS;
}
}// namespace

namespace Pupil::util {
bool BitmapTexture::Save(float *data, size_t w, size_t h, const char *save_path, FileFormat format) noexcept {
    switch (format) {
        case FileFormat::HDR:
            return SaveHdr(data, w, h, save_path);
        case FileFormat::EXR:
            return SaveExr(data, w, h, save_path);
    }
    return false;
}
}// namespace Pupil::util