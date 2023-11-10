#include "cemyuksel_hair.h"
#include "util/log.h"

#include <fstream>

namespace Pupil::resource {
    bool CyHair::Load(const char* path, CyHair& hair) noexcept {
        hair.header.file_info[87] = 0;

        std::ifstream in(path, std::ios::binary);

        if (!in.is_open()) {
            Pupil::Log::Warn("[{}] cannot be opened.", path);
            Pupil::Log::Warn("Cem Yuksel's hair file load failed.");
            return false;
        }

        in.read(reinterpret_cast<char*>(&hair.header), sizeof(hair.header));
        if (std::strncmp(hair.header.magic, "HAIR", 4) != 0) {
            Pupil::Log::Warn("[{}] format error.", path);
            Pupil::Log::Warn("Cem Yuksel's hair file load failed.");
            return false;
        }

        auto segments = std::vector<unsigned short>(hair.header.strands_num, hair.header.default_segments_num);
        if (hair.header.flags & 1) {// has segements
            in.read(reinterpret_cast<char*>(segments.data()), hair.header.strands_num * sizeof(unsigned short));
        }

        hair.strands_index.resize(segments.size() + 1);
        hair.strands_index[0] = 0;
        for (int i = 1; i < hair.strands_index.size(); ++i) {
            hair.strands_index[i] = hair.strands_index[i - 1] + 1 + segments[i - 1];
        }

        if (hair.header.flags & (1 << 1)) {// has points
            hair.positions.resize(hair.header.points_num * 3);
            in.read(reinterpret_cast<char*>(hair.positions.data()), hair.header.points_num * 3 * sizeof(float));
        } else {
            Pupil::Log::Warn("[{}] format error(has no points).", path);
            Pupil::Log::Warn("Cem Yuksel's hair file load failed.");
            return false;
        }

        float max_width = hair.header.default_thickness;
        hair.widths.resize(hair.header.points_num, hair.header.default_thickness);
        if (hair.header.flags & (1 << 2)) {// has thickness
            in.read(reinterpret_cast<char*>(hair.widths.data()), hair.header.points_num * sizeof(float));
            for (float width : hair.widths) max_width = std::max(max_width, width);
        }

        // if (hair.header.flags & (1 << 3)) {// has alpha
        // }
        // if (hair.header.flags & (1 << 4)) {// has color
        // }

        return true;
    }
}// namespace Pupil::resource