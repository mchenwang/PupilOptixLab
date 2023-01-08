#pragma once

#include <string>

namespace scene {
struct Transform {
    float matrix[16]{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    void Translate(float x, float y, float z) noexcept;
    // rotation axis is [ux, uy, uz]
    void Rotate(float ux, float uy, float uz, float angle) noexcept;
    void Scale(float x, float y, float z) noexcept;
};

enum class EShapeType {
    OBJ,
    PLY,
    SPHERE,
    RECTANGLE,
    CUBE
};

struct Shape {
    const EShapeType type;
    std::string filename{};
    bool face_normals = false;
    bool flip_tex_coords = false;
    bool flip_normals = false;

    Transform transform{};

    // for sphere
    float center[3]{ 0.f };
    float radius = 1.f;

    Shape(EShapeType) noexcept;
};
}// namespace scene