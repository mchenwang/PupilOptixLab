#pragma once

namespace Pupil::resource {
    struct Image {
        size_t w;
        size_t h;
        float* data;

        static bool Load(const char* path, Image& image, bool is_srgb = true) noexcept;
        static bool Save(const Image& image, const char* save_path) noexcept;
    };
}// namespace Pupil::resource