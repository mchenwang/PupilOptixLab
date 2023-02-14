#pragma once

#include "cuda_util/preprocessor.h"
#include "cuda_util/texture.h"
#include "scene/emitter.h"
#include "geometry.h"
#include "util.h"

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
#include <vector>

namespace scene {
class Scene;
}
#endif

namespace optix_util {
enum class EEmitterType {
    None,
    Triangle,
    Sphere
};

struct Emitter {
    EEmitterType type CONST_STATIC_INIT(EEmitterType::None);

    cuda::Texture radiance;

    float area;
    float select_probability;

    union {
        struct {
            struct {
                float3 pos;
                float3 normal;
                float2 tex;
            } v0, v1, v2;
        } triangle;
        struct {
            float3 center;
            float radius;
        } sphere;
    } geo;

    CUDA_HOSTDEVICE Emitter() noexcept {}

    CUDA_HOSTDEVICE static const Emitter &SelectOneEmiiter(float p, const cuda::ConstArrayView<Emitter> &emitters) noexcept {
        unsigned int i = 0;
        float sum_p = 0.f;
        for (; i < emitters.GetNum() - 1; ++i) {
            if (p > sum_p && p < sum_p + emitters[i].select_probability) {
                break;
            }
            sum_p += emitters[i].select_probability;
        }
        return emitters[i];
    }

    // CUDA_HOSTDEVICE float4 Eval(LocalGeometry hit_geo, float3 ray_dir) const noexcept {
    //     float3 emit_radiance = make_float3(0.f);
    //     float pdf = 0.f;
    //     float LNoL = dot(-ray_dir, hit_geo.normal);
    //     if (LNoL > 0.f) {
    //         emit_radiance = radiance.Sample(hit_geo.texcoord);
    //         pdf = 1.f / (area * LNoL);
    //     }
    //     return make_float4(emit_radiance, pdf);
    // }

    struct LocalRecord {
        float3 position;
        float3 normal;
        float3 radiance;
    };

    CUDA_HOSTDEVICE LocalRecord GetLocalInfo(const float3 p) const noexcept {
        LocalRecord ret;
        ret.position = p;
        switch (type) {
            case EEmitterType::Triangle: {
                float3 t = optix_util::GetBarycentricCoordinates(p, geo.triangle.v0.pos, geo.triangle.v1.pos, geo.triangle.v2.pos);
                ret.normal = geo.triangle.v0.normal * t.x + geo.triangle.v1.normal * t.y + geo.triangle.v2.normal * t.z;
                auto tex = geo.triangle.v0.tex * t.x + geo.triangle.v1.tex * t.y + geo.triangle.v2.tex * t.z;
                ret.radiance = radiance.Sample(tex);
            } break;
            case EEmitterType::Sphere: {
                ret.normal = (p - geo.sphere.center) / geo.sphere.radius;
                float2 tex = make_float2(asin(ret.normal.x) * M_1_PIf + 0.5f, asin(ret.normal.y) * M_1_PIf + 0.5f);
                ret.radiance = radiance.Sample(tex);
            } break;
        }
        ret.normal = normalize(ret.normal);
        return ret;
    }

    CUDA_HOSTDEVICE LocalRecord SampleDirect(const float u1, const float u2) const noexcept {
        LocalRecord ret;
        switch (type) {
            case EEmitterType::Triangle: {
                float3 t = optix_util::UniformSampleTriangle(u1, u2);
                ret.position = geo.triangle.v0.pos * t.x + geo.triangle.v1.pos * t.y + geo.triangle.v2.pos * t.z;
                ret.normal = geo.triangle.v0.normal * t.x + geo.triangle.v1.normal * t.y + geo.triangle.v2.normal * t.z;
                auto tex = geo.triangle.v0.tex * t.x + geo.triangle.v1.tex * t.y + geo.triangle.v2.tex * t.z;
                ret.radiance = radiance.Sample(tex);
            } break;
            case EEmitterType::Sphere: {
                float3 t = optix_util::UniformSampleSphere(u1, u2);
                ret.position = t * geo.sphere.radius + geo.sphere.center;
                ret.normal = t;
                float2 tex = make_float2(asin(t.x) * M_1_PIf + 0.5f, asin(t.y) * M_1_PIf + 0.5f);
                ret.radiance = radiance.Sample(tex);
            } break;
        }
        ret.normal = normalize(ret.normal);
        return ret;
    }
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
std::vector<Emitter> GenerateEmitters(scene::Scene *) noexcept;
#endif
}// namespace optix_util