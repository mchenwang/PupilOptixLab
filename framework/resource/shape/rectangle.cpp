#include "resource/shape.h"

namespace {
    // clang-format off

// XY-range [-1,1]x[-1,1]
float m_rect_positions[] = {
    -1.f, -1.f, 0.f,
     1.f, -1.f, 0.f,
     1.f,  1.f, 0.f,
    -1.f,  1.f, 0.f
};
float m_rect_normals[] = {
    0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f
};
float m_rect_texcoords[] = {
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
};
uint32_t m_rect_indices[] = { 0, 1, 2, 0, 2, 3 };

    // clang-format on
}// namespace

namespace Pupil::resource {
    util::CountableRef<Shape> Rectangle::Make(std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<Rectangle>(UserDisableTag{}, name));
    }

    Rectangle::Rectangle(UserDisableTag tag, std::string_view name) noexcept
        : TriangleMesh(tag, name) {
        m_flip_texcoord = false;
        SetVertex(m_rect_positions, 4);
        SetIndex(m_rect_indices, 2);
        SetNormal(m_rect_normals, 4);
        SetTexcoord(m_rect_texcoords, 4);

        aabb = util::AABB{{-1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}};
    }

    void* Rectangle::Clone() const noexcept {
        auto clone = new Rectangle(UserDisableTag{}, m_name);
        return clone;
    }

    uint64_t Rectangle::GetMemorySizeInByte() const noexcept {
        return sizeof(m_rect_indices) + sizeof(m_rect_positions) + sizeof(m_rect_normals) + sizeof(m_rect_texcoords);
    }

}// namespace Pupil::resource