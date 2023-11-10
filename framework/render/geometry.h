#pragma once

#include "cuda/util.h"
#include <vector_types.h>

#ifdef PUPIL_OPTIX
#include <optix.h>
#include "util.h"
#include "curve.h"
#endif

namespace Pupil::optix {
    struct TriMesh {
        bool                         flip_normals;
        bool                         flip_tex_coords;
        cuda::ConstArrayView<float3> positions;
        cuda::ConstArrayView<float3> normals;
        cuda::ConstArrayView<float2> texcoords;
        cuda::ConstArrayView<uint3>  indices;
    };

    struct Sphere {
        float3 center;
        float  radius;
        bool   flip_normal;
    };

    struct Curve {
        // cuda::ConstArrayView<float3> positions;
        // cuda::ConstArrayView<unsigned int> indices;
    };

    struct LocalGeometry {
        float3 position;
        float3 normal;
        float2 texcoord;
    };

    struct Geometry {
        enum class EType : unsigned int {
            TriMesh,
            Sphere,
            // for hair
            LinearBSpline,
            QuadraticBSpline,
            CubicBSpline,
            CatromSpline
        } type;

        union {
            TriMesh tri_mesh;
            Sphere  sphere;
            Curve   curve;
        };

        CUDA_HOSTDEVICE Geometry() noexcept {}

#ifdef PUPIL_OPTIX
        CUDA_DEVICE void GetHitLocalGeometry(LocalGeometry& ret) const noexcept {
            switch (type) {
                case EType::TriMesh: {
                    const auto face_index   = optixGetPrimitiveIndex();
                    const auto bary         = optixGetTriangleBarycentrics();
                    const auto [v0, v1, v2] = tri_mesh.indices[face_index];

                    const auto p0 = tri_mesh.positions[v0];
                    const auto p1 = tri_mesh.positions[v1];
                    const auto p2 = tri_mesh.positions[v2];
                    ret.position  = (1.f - bary.x - bary.y) * p0 + bary.x * p1 + bary.y * p2;
                    ret.position  = optixTransformPointFromObjectToWorldSpace(ret.position);

                    if (tri_mesh.normals) {
                        const auto n0 = tri_mesh.normals[v0];
                        const auto n1 = tri_mesh.normals[v1];
                        const auto n2 = tri_mesh.normals[v2];
                        ret.normal    = (1.f - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;
                    } else {
                        ret.normal = cross(p1 - p0, p2 - p0);
                    }
                    ret.normal = normalize(optixTransformNormalFromObjectToWorldSpace(ret.normal));

                    if (tri_mesh.flip_normals) ret.normal *= -1.f;

                    if (tri_mesh.texcoords) {
                        const auto t0 = tri_mesh.texcoords[v0];
                        const auto t1 = tri_mesh.texcoords[v1];
                        const auto t2 = tri_mesh.texcoords[v2];
                        ret.texcoord  = (1.f - bary.x - bary.y) * t0 + bary.x * t1 + bary.y * t2;
                        if (tri_mesh.flip_tex_coords) ret.texcoord.y = 1.f - ret.texcoord.y;
                    }
                } break;
                case EType::Sphere: {
                    ret.position         = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
                    const auto local_pos = optixTransformPointFromWorldToObjectSpace(ret.position);
                    ret.texcoord         = Pupil::optix::GetSphereTexcoord(normalize(local_pos - sphere.center));
                    ret.normal           = normalize(optixTransformNormalFromObjectToWorldSpace(local_pos - sphere.center));
                    if (sphere.flip_normal) ret.normal *= -1.f;
                } break;
                case EType::LinearBSpline: {
                    ret.position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

                    const auto u = optixGetCurveParameter();

                    const auto gas         = optixGetGASTraversableHandle();
                    const auto prim_idx    = optixGetPrimitiveIndex();
                    const auto gas_sbt_idx = optixGetSbtGASIndex();
                    float4     ctrl_points[2];

                    optixGetLinearCurveVertexData(gas, prim_idx, gas_sbt_idx, 0.0f, ctrl_points);

                    LinearInterpolator interpolator;
                    interpolator.Initialize(ctrl_points);

                    float3 local_pos = optixTransformPointFromWorldToObjectSpace(ret.position);
                    ret.normal       = SurfaceNormal(interpolator, u, local_pos);
                    ret.normal       = optixTransformNormalFromObjectToWorldSpace(ret.normal);

                    float3 tangent = CurveTangent(interpolator, u);
                    float3 y       = normalize(cross(optixGetWorldRayDirection(), tangent));
                    float  v       = dot(y, ret.normal);

                    ret.normal   = tangent;
                    ret.texcoord = make_float2(u, v);
                } break;
                case EType::QuadraticBSpline: {
                    ret.position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

                    const auto u = optixGetCurveParameter();

                    const auto gas         = optixGetGASTraversableHandle();
                    const auto prim_idx    = optixGetPrimitiveIndex();
                    const auto gas_sbt_idx = optixGetSbtGASIndex();
                    float4     ctrl_points[3];

                    optixGetQuadraticBSplineVertexData(gas, prim_idx, gas_sbt_idx, 0.0f, ctrl_points);

                    QuadraticInterpolator interpolator;
                    interpolator.InitializeFromBSpline(ctrl_points);

                    float3 local_pos = optixTransformPointFromWorldToObjectSpace(ret.position);
                    ret.normal       = SurfaceNormal(interpolator, optixGetCurveParameter(), local_pos);
                    ret.normal       = optixTransformNormalFromObjectToWorldSpace(ret.normal);

                    float3 tangent = CurveTangent(interpolator, optixGetCurveParameter());
                    float3 y       = normalize(cross(optixGetWorldRayDirection(), tangent));
                    float  v       = dot(y, ret.normal);

                    ret.normal   = tangent;
                    ret.texcoord = make_float2(u, v);
                } break;
                case EType::CubicBSpline: {
                    ret.position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

                    const auto u = optixGetCurveParameter();

                    const auto gas         = optixGetGASTraversableHandle();
                    const auto prim_idx    = optixGetPrimitiveIndex();
                    const auto gas_sbt_idx = optixGetSbtGASIndex();
                    float4     ctrl_points[4];

                    optixGetCubicBSplineVertexData(gas, prim_idx, gas_sbt_idx, 0.0f, ctrl_points);

                    CubicInterpolator interpolator;
                    interpolator.InitializeFromBSpline(ctrl_points);

                    float3 local_pos = optixTransformPointFromWorldToObjectSpace(ret.position);
                    ret.normal       = SurfaceNormal(interpolator, optixGetCurveParameter(), local_pos);
                    ret.normal       = optixTransformNormalFromObjectToWorldSpace(ret.normal);

                    float3 tangent = CurveTangent(interpolator, optixGetCurveParameter());
                    float3 y       = normalize(cross(optixGetWorldRayDirection(), tangent));
                    float  v       = dot(y, ret.normal);

                    ret.normal   = tangent;
                    ret.texcoord = make_float2(u, v);
                } break;
                case EType::CatromSpline: {
                    ret.position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

                    const auto u = optixGetCurveParameter();

                    const auto gas         = optixGetGASTraversableHandle();
                    const auto prim_idx    = optixGetPrimitiveIndex();
                    const auto gas_sbt_idx = optixGetSbtGASIndex();
                    float4     ctrl_points[4];

                    optixGetCubicBSplineVertexData(gas, prim_idx, gas_sbt_idx, 0.0f, ctrl_points);

                    CubicInterpolator interpolator;
                    interpolator.InitializeFromCatrom(ctrl_points);

                    float3 local_pos = optixTransformPointFromWorldToObjectSpace(ret.position);
                    ret.normal       = SurfaceNormal(interpolator, optixGetCurveParameter(), local_pos);
                    ret.normal       = optixTransformNormalFromObjectToWorldSpace(ret.normal);

                    float3 tangent = CurveTangent(interpolator, optixGetCurveParameter());
                    float3 y       = normalize(cross(optixGetWorldRayDirection(), tangent));
                    float  v       = dot(y, ret.normal);

                    ret.normal   = tangent;
                    ret.texcoord = make_float2(u, v);
                } break;
            }
        }

        CUDA_DEVICE void GetHitLocalGeometry(LocalGeometry& ret, float3 ray_dir, bool twosided) const noexcept {
            GetHitLocalGeometry(ret);
            if (dot(-ray_dir, ret.normal) < 0.f && twosided)
                ret.normal = -ret.normal;
        }
#endif
    };
}// namespace Pupil::optix