#include "resource/shape.h"

namespace {
    // clang-format off

// XYZ-range [-1,-1,-1]x[1,1,1]
float m_cube_positions[] = {
    -1.f,-1.f,-1.f, -1.f,-1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f,-1.f,
     1.f,-1.f,-1.f, -1.f,-1.f,-1.f, -1.f, 1.f,-1.f,  1.f, 1.f,-1.f,
     1.f,-1.f, 1.f,  1.f,-1.f,-1.f,  1.f, 1.f,-1.f,  1.f, 1.f, 1.f,
    -1.f,-1.f, 1.f,  1.f,-1.f, 1.f,  1.f, 1.f, 1.f, -1.f, 1.f, 1.f,
    -1.f, 1.f, 1.f,  1.f, 1.f, 1.f,  1.f, 1.f,-1.f, -1.f, 1.f,-1.f,
    -1.f,-1.f,-1.f,  1.f,-1.f,-1.f,  1.f,-1.f, 1.f, -1.f,-1.f, 1.f
};
float m_cube_normals[] = {
    -1.f,0.f,0.f, -1.f,0.f,0.f, -1.f,0.f,0.f, -1.f,0.f,0.f,
    0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f,
    1.f,0.f,0.f, 1.f,0.f,0.f, 1.f,0.f,0.f, 1.f,0.f,0.f,
    0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f,
    0.f,1.f,0.f, 0.f,1.f,0.f, 0.f,1.f,0.f, 0.f,1.f,0.f,
    0.f,-1.f,0.f, 0.f,-1.f,0.f, 0.f,-1.f,0.f, 0.f,-1.f,0.f
};
float m_cube_texcoords[] = {
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f
};
uint32_t m_cube_indices[] = {
    0,1,2, 0,2,3,
    4,5,6, 4,6,7,
    8,9,10, 8,10,11,
    12,13,14, 12,14,15,
    16,17,18, 16,18,19,
    20,21,22, 20,22,23
};
    // clang-format on
}// namespace

namespace Pupil::resource {
    util::CountableRef<Shape> Cube::Make(std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<Cube>(UserDisableTag{}, name));
    }

    Cube::Cube(UserDisableTag tag, std::string_view name) noexcept
        : TriangleMesh(tag, name) {
        m_flip_texcoord = false;
        SetVertex(m_cube_positions, 24);
        SetIndex(m_cube_indices, 12);
        SetNormal(m_cube_normals, 24);
        SetTexcoord(m_cube_texcoords, 24);

        aabb = util::AABB{{-1.f, -1.f, -1.f}, {1.f, 1.f, 1.f}};
    }

    void* Cube::Clone() const noexcept {
        auto clone = new Cube(UserDisableTag{}, m_name);
        return clone;
    }

    uint64_t Cube::GetMemorySizeInByte() const noexcept {
        return sizeof(m_cube_indices) + sizeof(m_cube_positions) + sizeof(m_cube_normals) + sizeof(m_cube_texcoords);
    }
}// namespace Pupil::resource