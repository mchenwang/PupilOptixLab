#include "transform.h"

#include <cmath>
#include <DirectXMath.h>

namespace Pupil::util {
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

    // float rotate[16]{
    //     1.f - 2.f * c * c - 2.f * d * d, 2.f * b * c - 2.f * a * d, 2.f * a * c + 2.f * b * d, 0.f,
    //     2.f * b * c + 2.f * a * d, 1.f - 2.f * b * b - 2.f * d * d, 2.f * c * d - 2.f * a * b, 0.f,
    //     2.f * b * d - 2.f * a * c, 2.f * a * b + 2.f * c * d, 1.f - 2.f * b * b - 2.f * c * c, 0.f,
    //     0.f, 0.f, 0.f, 1.f
    // };
    // MatrixMultiply(rotate, matrix.e, matrix.e);

    DirectX::XMMATRIX rotate{
        1.f - 2.f * c * c - 2.f * d * d, 2.f * b * c - 2.f * a * d, 2.f * a * c + 2.f * b * d, 0.f,
        2.f * b * c + 2.f * a * d, 1.f - 2.f * b * b - 2.f * d * d, 2.f * c * d - 2.f * a * b, 0.f,
        2.f * b * d - 2.f * a * c, 2.f * a * b + 2.f * c * d, 1.f - 2.f * b * b - 2.f * c * c, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    matrix = rotate * matrix;
}

void Transform::Translate(float x, float y, float z) noexcept {
    // float translate[16]{
    //     1.f, 0.f, 0.f, x,
    //     0.f, 1.f, 0.f, y,
    //     0.f, 0.f, 1.f, z,
    //     0.f, 0.f, 0.f, 1.f
    // };
    // MatrixMultiply(translate, matrix.e, matrix.e);

    DirectX::XMMATRIX translate{
        1.f, 0.f, 0.f, x,
        0.f, 1.f, 0.f, y,
        0.f, 0.f, 1.f, z,
        0.f, 0.f, 0.f, 1.f
    };
    matrix = translate * matrix;
}

void Transform::Scale(float x, float y, float z) noexcept {
    // float scale[16]{
    //     x, 0.f, 0.f, 0.f,
    //     0.f, y, 0.f, 0.f,
    //     0.f, 0.f, z, 0.f,
    //     0.f, 0.f, 0.f, 1.f
    // };
    // MatrixMultiply(scale, matrix.e, matrix.e);
    DirectX::XMMATRIX scale{
        x, 0.f, 0.f, 0.f,
        0.f, y, 0.f, 0.f,
        0.f, 0.f, z, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    matrix = scale * matrix;
}

void Transform::LookAt(const Float3 &origin, const Float3 &target, const Float3 &up) noexcept {
    DirectX::XMFLOAT3 t_eye_position{ origin.x, origin.y, origin.z };
    DirectX::XMVECTOR eye_position = DirectX::XMLoadFloat3(&t_eye_position);
    DirectX::XMFLOAT3 t_focus_position{ target.x, target.y, target.z };
    DirectX::XMVECTOR focus_position = DirectX::XMLoadFloat3(&t_focus_position);
    DirectX::XMFLOAT3 t_up_direction{ up.x, up.y, up.z };
    DirectX::XMVECTOR up_direction = DirectX::XMLoadFloat3(&t_up_direction);

    auto world_to_camera = DirectX::XMMatrixLookAtRH(eye_position, focus_position, up_direction);
    auto camera_to_world = DirectX::XMMatrixTranspose(DirectX::XMMatrixInverse(nullptr, world_to_camera));
    matrix = camera_to_world;
}

Float3 Transform::TransformPoint(const Float3 point, const Mat4 &transform_matrix) noexcept {
    auto &m = transform_matrix.e;
    float x = m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3];
    float y = m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7];
    float z = m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11];
    float w = m[12] * point.x + m[13] * point.y + m[14] * point.z + m[15];
    return Float3{ x / w, y / w, z / w };
}
Float3 Transform::TransformVector(const Float3 vector, const Mat4 &transform_matrix) noexcept {
    auto &m = transform_matrix.e;
    float x = m[0] * vector.x + m[1] * vector.y + m[2] * vector.z;
    float y = m[4] * vector.x + m[5] * vector.y + m[6] * vector.z;
    float z = m[8] * vector.x + m[9] * vector.y + m[10] * vector.z;
    return Float3{ x, y, z };
}
Float3 Transform::TransformNormal(const Float3 normal, const Mat4 &transform_matrix_inv_t) noexcept {
    auto &m = transform_matrix_inv_t.e;
    float x = m[0] * normal.x + m[1] * normal.y + m[2] * normal.z;
    float y = m[4] * normal.x + m[5] * normal.y + m[6] * normal.z;
    float z = m[8] * normal.x + m[9] * normal.y + m[10] * normal.z;

    float len = std::sqrtf(x * x + y * y + z * z);
    return Float3{ x / len, y / len, z / len };
}
}// namespace Pupil::util