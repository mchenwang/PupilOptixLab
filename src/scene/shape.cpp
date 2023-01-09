#include "shape.h"
#include <cmath>

namespace {
// clang-format off
void MatrixMultiply(float l[16], float r[16], float *ans) noexcept {
    float temp[16]{};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i * 4 + j] = l[i * 4 + 0] * r[0 * 4 + j] 
                            + l[i * 4 + 1] * r[1 * 4 + j] 
                            + l[i * 4 + 2] * r[2 * 4 + j] 
                            + l[i * 4 + 3] * r[3 * 4 + j];
        }
    }

    memcpy(ans, temp, sizeof(float) * 16);
}
// clang-format on
}// namespace

namespace scene {

void Transform::Rotate(float ux, float uy, float uz, float angle) noexcept {
    /*
        Target vector v
        Rotation axis Vector u = normalize([ux, uy, uz]^T)
        Rotation angle \theta = angle / 180 * PI
        Quaternion v = [0, v]
        Quaternion q = [cos(0.5 * \theta), sin(0.5 * \theta) * u]

        a = cos(0.5 * \theta)
        b = sin(0.5 * \theta)u_x
        c = sin(0.5 * \theta)u_y
        d = sin(0.5 * \theta)u_z
        qvq* = [1   0           0           0
                0   1-2cc-2dd   2bc-2ad     2ac+2bd
                0   2bc+2ad     1-2bb-2dd   2cd-2ab
                0   2bd-2ac     2ab+2cd     1-2bb-2cc] v
    */

    float u_len = std::sqrtf(ux * ux + uy * uy + uz * uz);
    ux /= u_len, uy /= u_len, uz /= u_len;

    float theta = angle / 180.f * 3.14159265358979323846f;

    float a = cos(0.5f * theta);
    float b = sin(0.5f * theta) * ux;
    float c = sin(0.5f * theta) * uy;
    float d = sin(0.5f * theta) * uz;

    float rotate[16]{
        1.f - 2.f * c * c - 2.f * d * d, 2.f * b * c - 2.f * a * d, 2.f * a * c + 2.f * b * d, 0.f,
        2.f * b * c + 2.f * a * d, 1.f - 2.f * b * b - 2.f * d * d, 2.f * c * d - 2.f * a * b, 0.f,
        2.f * b * d - 2.f * a * c, 2.f * a * b + 2.f * c * d, 1.f - 2.f * b * b - 2.f * c * c, 0.f,
        0.f, 0.f, 0.f, 1.f
    };

    MatrixMultiply(rotate, matrix, matrix);
}

void Transform::Translate(float x, float y, float z) noexcept {
    float translate[16]{
        1.f, 0.f, 0.f, x,
        0.f, 1.f, 0.f, y,
        0.f, 0.f, 1.f, z,
        0.f, 0.f, 0.f, 1.f
    };
    MatrixMultiply(translate, matrix, matrix);
}

void Transform::Scale(float x, float y, float z) noexcept {
    float scale[16]{
        x, 0.f, 0.f, 0.f,
        0.f, y, 0.f, 0.f,
        0.f, 0.f, z, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    MatrixMultiply(scale, matrix, matrix);
}

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