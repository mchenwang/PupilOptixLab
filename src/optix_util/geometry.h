#pragma once

#include "cuda_util/data_view.h"
#include <vector_types.h>

#if defined(__CUDACC__) || defined(__CUDABE__)
#include <optix.h>
#include "cuda_util/vec_math.h"
#endif

#if !defined(__CUDACC__) && !defined(__CUDABE__)
#include "scene/shape.h"
#endif

namespace optix_util {
struct TriMesh {
    cuda::ConstArrayView<float3> positions;
    cuda::ConstArrayView<float3> normals;
    cuda::ConstArrayView<float2> texcoords;
    cuda::ConstArrayView<uint3> indices;
    bool flip_normals;
    bool flip_tex_coords;
};

struct Sphere {
    float3 center;
    float radius;
    bool flip_normal;
};

struct LocalGeometry {
    float3 position;
    float3 normal;
    float2 texcoord;
};

struct Geometry {
    enum class EType {
        TriMesh,
        Sphere
    } type;

    union {
        TriMesh tri_mesh;
        Sphere sphere;
    };

    CUDA_HOSTDEVICE Geometry() noexcept {}

#if defined(__CUDACC__) || defined(__CUDABE__)
    CUDA_HOSTDEVICE LocalGeometry GetHitLocalGeometry() const noexcept {
        LocalGeometry ret;
        switch (type) {
            case EType::TriMesh: {
                const auto face_index = optixGetPrimitiveIndex();
                const auto bary = optixGetTriangleBarycentrics();
                // const auto vertex_index = make_uint3(face_index * 3 + 0, face_index * 3 + 1, face_index * 3 + 2);
                const auto [v0, v1, v2] = tri_mesh.indices[face_index];

                const auto p0 = tri_mesh.positions[v0];
                const auto p1 = tri_mesh.positions[v1];
                const auto p2 = tri_mesh.positions[v2];
                ret.position = (1.f - bary.x - bary.y) * p0 + bary.x * p1 + bary.y * p2;
                // ret.position = optixTransformPointFromObjectToWorldSpace(ret.position);

                if (tri_mesh.normals) {
                    const auto n0 = tri_mesh.normals[v0];
                    const auto n1 = tri_mesh.normals[v1];
                    const auto n2 = tri_mesh.normals[v2];
                    ret.normal = (1.f - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;
                    ret.normal = normalize(ret.normal);
                } else {
                    ret.normal = normalize(cross(p1 - p0, p2 - p0));
                }
                if (tri_mesh.flip_normals) ret.normal *= -1.f;

                if (tri_mesh.texcoords) {
                    const auto t0 = tri_mesh.texcoords[v0];
                    const auto t1 = tri_mesh.texcoords[v1];
                    const auto t2 = tri_mesh.texcoords[v2];
                    ret.texcoord = (1.f - bary.x - bary.y) * t0 + bary.x * t1 + bary.y * t2;
                    if (tri_mesh.flip_tex_coords) ret.texcoord.y = 1.f - ret.texcoord.y;
                }
            } break;
            case EType::Sphere: {
                const auto p_w = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
                ret.position = optixTransformPointFromWorldToObjectSpace(p_w);
                ret.normal = normalize(ret.position - sphere.center);
                if (sphere.flip_normal) ret.normal *= -1.f;
            } break;
        }
        return ret;
    }
#endif

#if !defined(__CUDACC__) && !defined(__CUDABE__)
    void LoadGeometry(const scene::Shape &) noexcept;
#endif
};
}// namespace optix_util