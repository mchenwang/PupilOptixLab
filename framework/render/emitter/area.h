#pragma once

#include "cuda/data_view.h"
#include "render/texture.h"

namespace Pupil::optix {
    struct TriMeshEmitter {
        Texture                      radiance;
        cuda::ConstArrayView<uint3>  indices;
        cuda::ConstArrayView<float3> pos_ws;
        cuda::ConstArrayView<float3> nor_ws;
        cuda::ConstArrayView<float2> tex_ls;
        cuda::ConstArrayView<float>  areas;
        float                        inv_num;

        CUDA_DEVICE void SampleDirect(EmitterSampleRecord& ret, LocalGeometry& hit_geo, float2 xi) const noexcept {
            unsigned int sub_face_index = xi.x * indices.GetNum();

            float cdf = sub_face_index * 1.f / indices.GetNum();
            xi.x      = (xi.x - cdf) * indices.GetNum();

            auto area         = areas[sub_face_index];
            auto [i1, i2, i3] = indices[sub_face_index];
            float3 pos1 = pos_ws[i1], nor1 = nor_ws[i1];
            float3 pos2 = pos_ws[i2], nor2 = nor_ws[i2];
            float3 pos3 = pos_ws[i3], nor3 = nor_ws[i3];

            float2 tex1 = tex_ls[i1];
            float2 tex2 = tex_ls[i2];
            float2 tex3 = tex_ls[i3];

            float3 t        = Pupil::optix::UniformSampleTriangle(xi.x, xi.y);
            float3 position = pos1 * t.x + pos2 * t.y + pos3 * t.z;
            float3 normal   = normalize(nor1 * t.x + nor2 * t.y + nor3 * t.z);
            auto   tex      = tex1 * t.x + tex2 * t.y + tex3 * t.z;
            ret.radiance    = radiance.Sample(tex);

            ret.wi     = normalize(position - hit_geo.position);
            float NoL  = dot(hit_geo.normal, ret.wi);
            float LNoL = dot(normal, -ret.wi);
            if (NoL > 0.f && LNoL > 0.f) {
                float distance = length(position - hit_geo.position);
                ret.pdf        = distance * distance / (LNoL * area) * inv_num;
                ret.distance   = distance;
            }

            ret.pos    = position;
            ret.normal = normal;
        }

        CUDA_DEVICE void Eval(EmitEvalRecord& ret, LocalGeometry& emit_local_geo, float3 scatter_pos) const noexcept {
            auto   area = areas[ret.primitive_index];
            float3 dir  = normalize(scatter_pos - emit_local_geo.position);
            float  LNoL = dot(emit_local_geo.normal, dir);
            if (LNoL > 0.f) {
                float distance = length(scatter_pos - emit_local_geo.position);
                ret.pdf        = distance * distance / (LNoL * area) * inv_num;
                ret.radiance   = radiance.Sample(emit_local_geo.texcoord);
            }
        }
    };
}// namespace Pupil::optix