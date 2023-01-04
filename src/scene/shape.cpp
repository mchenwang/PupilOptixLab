#include "shape.h"

namespace scene {
Shape::Shape(EShapeType type) noexcept : type(type) {
    switch (type) {
        case EShapeType::OBJ:
            face_normals = false;
            flip_tex_coords = true;
            flip_normals = false;
            break;
        case EShapeType::PLY:
            face_normals = false;
            flip_tex_coords = false;
            flip_normals = false;
            break;
        case EShapeType::SPHERE:
            face_normals = false;
            flip_tex_coords = false;
            flip_normals = false;
            center[0] = center[1] = center[2] = 0.f;
            radius = 1.f;
            break;
        case EShapeType::CUBE:
            face_normals = true;
            flip_tex_coords = false;
            flip_normals = false;
            break;
        case EShapeType::RECTANGLE:
            face_normals = true;
            flip_tex_coords = false;
            flip_normals = false;
            break;
        default:
            break;
    }
}
}// namespace scene