#pragma once

#include <vector>

namespace Pupil::resource {
    struct CyHair {
        struct Header {
            // Bytes 0 - 3  Must be "HAIR" in ascii code(48 41 49 52)
            char magic[4];
            // Bytes 4 - 7  Number of hair strands as unsigned int
            uint32_t strands_num;
            // Bytes 8 - 11  Total number of points of all strands as unsigned int
            uint32_t points_num;
            // Bytes 12 - 15  Bit array of data in the file
            // Bit - 5 to Bit - 31 are reserved for future extension(must be 0).
            uint32_t flags;
            // Bytes 16 - 19  Default number of segments of hair strands as unsigned int
            // If the file does not have a segments array, this default value is used.
            uint32_t default_segments_num;
            // Bytes 20 - 23  Default thickness hair strands as float
            // If the file does not have a thickness array, this default value is used.
            float default_thickness;
            // Bytes 24 - 27  Default transparency hair strands as float
            // If the file does not have a transparency array, this default value is used.
            float default_alpha;
            // Bytes 28 - 39  Default color hair strands as float array of size 3
            // If the file does not have a color array, this default value is used.
            float default_color[3];
            // Bytes 40 - 127  File information as char array of size 88 in ascii
            char file_info[88];
        };

        Header                header;
        std::vector<uint32_t> strands_index;
        std::vector<float>    positions;
        std::vector<float>    widths;

        static bool Load(const char* path, CyHair& hair) noexcept;
    };
}// namespace Pupil::resource