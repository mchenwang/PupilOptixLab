#include "emitter.h"
#include "scene/scene.h"
#include "device/cuda_texture.h"

#include <DirectXMath.h>

namespace {
float SplitMesh(std::vector<optix_util::Emitter> &emitters,
                uint32_t vertex_num, uint32_t face_num, uint32_t *indices,
                const float *positions, const float *normals, const float *texcoords,
                const util::Transform &transform, const cuda::Texture &radiance, const float select_wight) noexcept {
    DirectX::XMMATRIX dx_transform(transform.matrix);
    DirectX::XMFLOAT4X4 dx_tra_inv_t;
    DirectX::XMStoreFloat4x4(
        &dx_tra_inv_t,
        DirectX::XMMatrixTranspose(
            DirectX::XMMatrixInverse(nullptr, dx_transform)));
    float *tra_inv_t = reinterpret_cast<float *>(dx_tra_inv_t.m);

    float weight_sum = 0.f;

    for (auto i = 0u; i < face_num; ++i) {
        optix_util::Emitter emitter;
        emitter.type = optix_util::EEmitterType::Triangle;

        auto idx0 = indices[i * 3 + 0];
        auto idx1 = indices[i * 3 + 1];
        auto idx2 = indices[i * 3 + 2];

        util::float3 p0(positions[idx0 * 3 + 0], positions[idx0 * 3 + 1], positions[idx0 * 3 + 2]);
        util::float3 p1(positions[idx1 * 3 + 0], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
        util::float3 p2(positions[idx2 * 3 + 0], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);

        p0 = util::Transform::TransformPoint(p0, transform.matrix);
        p1 = util::Transform::TransformPoint(p1, transform.matrix);
        p2 = util::Transform::TransformPoint(p2, transform.matrix);

        util::float3 n0(normals[idx0 * 3 + 0], normals[idx0 * 3 + 1], normals[idx0 * 3 + 2]);
        util::float3 n1(normals[idx1 * 3 + 0], normals[idx1 * 3 + 1], normals[idx1 * 3 + 2]);
        util::float3 n2(normals[idx2 * 3 + 0], normals[idx2 * 3 + 1], normals[idx2 * 3 + 2]);

        n0 = util::Transform::TransformNormal(n0, tra_inv_t);
        n1 = util::Transform::TransformNormal(n1, tra_inv_t);
        n2 = util::Transform::TransformNormal(n2, tra_inv_t);

        emitter.geo.triangle.v0.pos = make_float3(p0.x, p0.y, p0.z);
        emitter.geo.triangle.v0.normal = make_float3(n0.x, n0.y, n0.z);
        emitter.geo.triangle.v0.tex = make_float2(texcoords[idx0 * 2 + 0], texcoords[idx0 * 2 + 1]);

        emitter.geo.triangle.v1.pos = make_float3(p1.x, p1.y, p1.z);
        emitter.geo.triangle.v1.normal = make_float3(n1.x, n1.y, n1.z);
        emitter.geo.triangle.v1.tex = make_float2(texcoords[idx1 * 2 + 0], texcoords[idx1 * 2 + 1]);

        emitter.geo.triangle.v2.pos = make_float3(p2.x, p2.y, p2.z);
        emitter.geo.triangle.v2.normal = make_float3(n2.x, n2.y, n2.z);
        emitter.geo.triangle.v2.tex = make_float2(texcoords[idx2 * 2 + 0], texcoords[idx2 * 2 + 1]);

        auto v1 = emitter.geo.triangle.v1.pos - emitter.geo.triangle.v0.pos;
        auto v2 = emitter.geo.triangle.v2.pos - emitter.geo.triangle.v0.pos;
        emitter.area = length(cross(v1, v2)) * 0.5f;

        emitter.radiance = radiance;
        emitter.select_probability = select_wight * emitter.area;

        weight_sum += emitter.select_probability;

        emitters.emplace_back(emitter);
    }

    return weight_sum;
}

inline float GetMax(float r, float g, float b) noexcept {
    return (r > g ? (r > b ? r : b) : (g > b ? g : b));
}

float GetWeight(util::Texture texture) noexcept {
    float w = 0.f;
    switch (texture.type) {
        case util::ETextureType::RGB:
            w = GetMax(texture.rgb.color.r, texture.rgb.color.g, texture.rgb.color.b);
            break;
        case util::ETextureType::Checkerboard: {
            float p1_w = GetMax(texture.checkerboard.patch1.r, texture.checkerboard.patch1.g, texture.checkerboard.patch1.b);
            float p2_w = GetMax(texture.checkerboard.patch2.r, texture.checkerboard.patch2.g, texture.checkerboard.patch2.b);
            w = (p1_w + p2_w) * 0.5f;
        } break;
        case util::ETextureType::Bitmap: {
            for (size_t i = 0; i < texture.bitmap.w; i++) {
                for (size_t j = 0; j < texture.bitmap.h; j++) {
                    w += GetMax(
                        texture.bitmap.data[(i * texture.bitmap.w + j) * 4 + 0],
                        texture.bitmap.data[(i * texture.bitmap.w + j) * 4 + 1],
                        texture.bitmap.data[(i * texture.bitmap.w + j) * 4 + 2]);
                }
            }
            w /= 1.f * texture.bitmap.w * texture.bitmap.h;
        } break;
    }
    return w;
}
}// namespace

namespace optix_util {
std::vector<Emitter> GenerateEmitters(scene::Scene *scene) noexcept {
    std::vector<Emitter> emitters;
    auto tex_mngr = util::Singleton<device::CudaTextureManager>::instance();
    float select_weight_sum = 0.f;

    for (auto &&shape : scene->shapes) {
        if (!shape.is_emitter) continue;

        auto radiance = tex_mngr->GetCudaTexture(shape.emitter.area.radiance);
        float select_weight = GetWeight(shape.emitter.area.radiance);

        size_t pre_emitters_num = emitters.size();

        switch (shape.type) {
            case scene::EShapeType::_cube: {
                select_weight_sum += SplitMesh(emitters, shape.cube.vertex_num, shape.cube.face_num, shape.cube.indices,
                                               shape.cube.positions, shape.cube.normals, shape.cube.texcoords,
                                               shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_obj: {
                select_weight_sum += SplitMesh(emitters, shape.obj.vertex_num, shape.obj.face_num, shape.obj.indices,
                                               shape.obj.positions, shape.obj.normals, shape.obj.texcoords,
                                               shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_rectangle: {
                select_weight_sum += SplitMesh(emitters, shape.rect.vertex_num, shape.rect.face_num, shape.rect.indices,
                                               shape.rect.positions, shape.rect.normals, shape.rect.texcoords,
                                               shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_sphere: {
                optix_util::Emitter emitter;
                emitter.type = optix_util::EEmitterType::Sphere;
                util::float3 o(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                util::float3 p(o.x + shape.sphere.radius, o.y, o.z);
                o = util::Transform::TransformPoint(o, shape.transform.matrix);
                p = util::Transform::TransformPoint(p, shape.transform.matrix);

                emitter.geo.sphere.center = make_float3(o.x, o.y, o.z);
                emitter.geo.sphere.radius = length(emitter.geo.sphere.center - make_float3(p.x, p.y, p.z));
                emitter.area = 4 * 3.14159265358979323846f * emitter.geo.sphere.radius * emitter.geo.sphere.radius;
                emitter.radiance = radiance;
                emitter.select_probability = select_weight * emitter.area;

                select_weight_sum += emitter.select_probability;

                emitters.emplace_back(emitter);
            } break;
        }

        shape.sub_emitters_num = static_cast<unsigned int>(
            emitters.size() - pre_emitters_num);
    }

    float pre_weight = 0.f;
    for (auto &&e : emitters) {
        // auto t_p = e.select_probability;
        // e.select_probability += pre_weight;
        e.select_probability /= select_weight_sum;
        // pre_weight += t_p;
    }

    return emitters;
}
}// namespace optix_util