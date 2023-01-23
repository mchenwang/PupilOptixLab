#include "camera.h"
#include "vec_math.h"

#include <DirectXMath.h>

namespace cuda {
void Camera::SetCameraTransform(float fov, float aspect_ratio, float near_clip, float far_clip) noexcept {
    auto dxm_camera_to_sample =
        DirectX::XMMatrixScaling(0.5f, 0.5f, 1.f) *
        DirectX::XMMatrixTranslation(1.f, 1.f, 0.f) *
        DirectX::XMMatrixPerspectiveFovRH(fov / 180.f * 3.14159265358979323846f, aspect_ratio, near_clip, far_clip);

    auto dxm_sample_to_camera = DirectX::XMMatrixInverse(nullptr, dxm_camera_to_sample);
    DirectX::XMFLOAT4X4 temp;
    DirectX::XMStoreFloat4x4(&temp, dxm_sample_to_camera);
    sample_to_camera.r0 = make_float4(temp.m[0][0], temp.m[0][1], temp.m[0][2], temp.m[0][3]);
    sample_to_camera.r1 = make_float4(temp.m[1][0], temp.m[1][1], temp.m[1][2], temp.m[1][3]);
    sample_to_camera.r2 = make_float4(temp.m[2][0], temp.m[2][1], temp.m[2][2], temp.m[2][3]);
    sample_to_camera.r3 = make_float4(temp.m[3][0], temp.m[3][1], temp.m[3][2], temp.m[3][3]);
}

void Camera::SetWorldTransform(float matrix[16]) noexcept {
    camera_to_world.r0 = make_float4(matrix[0], matrix[1], matrix[2], matrix[3]);
    camera_to_world.r1 = make_float4(matrix[4], matrix[5], matrix[6], matrix[7]);
    camera_to_world.r2 = make_float4(matrix[8], matrix[9], matrix[10], matrix[11]);
    camera_to_world.r3 = make_float4(matrix[12], matrix[13], matrix[14], matrix[15]);
}
}// namespace cuda