#include "emitter.h"
#include "scene/scene.h"
#include "cuda/texture.h"
#include "cuda/util.h"

#include "system/type.h"

#include <DirectXMath.h>

namespace {
using namespace Pupil;

float SplitMesh(std::vector<Pupil::optix::Emitter> &emitters,
                uint32_t vertex_num, uint32_t face_num, uint32_t *indices,
                const float *positions, const float *normals, const float *texcoords,
                const util::Transform &transform, const cuda::Texture &radiance, const float select_wight) noexcept {
    auto normal_transform = transform.matrix.GetInverse().GetTranspose();

    float weight_sum = 0.f;

    for (auto i = 0u; i < face_num; ++i) {
        Pupil::optix::Emitter emitter;
        emitter.type = Pupil::optix::EEmitterType::TriArea;

        auto idx0 = indices[i * 3 + 0];
        auto idx1 = indices[i * 3 + 1];
        auto idx2 = indices[i * 3 + 2];

        util::Float3 p0(positions[idx0 * 3 + 0], positions[idx0 * 3 + 1], positions[idx0 * 3 + 2]);
        util::Float3 p1(positions[idx1 * 3 + 0], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
        util::Float3 p2(positions[idx2 * 3 + 0], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);

        p0 = util::Transform::TransformPoint(p0, transform.matrix);
        p1 = util::Transform::TransformPoint(p1, transform.matrix);
        p2 = util::Transform::TransformPoint(p2, transform.matrix);

        util::Float3 n0(normals[idx0 * 3 + 0], normals[idx0 * 3 + 1], normals[idx0 * 3 + 2]);
        util::Float3 n1(normals[idx1 * 3 + 0], normals[idx1 * 3 + 1], normals[idx1 * 3 + 2]);
        util::Float3 n2(normals[idx2 * 3 + 0], normals[idx2 * 3 + 1], normals[idx2 * 3 + 2]);

        n0 = util::Transform::TransformNormal(n0, normal_transform);
        n1 = util::Transform::TransformNormal(n1, normal_transform);
        n2 = util::Transform::TransformNormal(n2, normal_transform);

        emitter.area.geo.v0.pos = make_float3(p0.x, p0.y, p0.z);
        emitter.area.geo.v0.normal = make_float3(n0.x, n0.y, n0.z);
        emitter.area.geo.v0.tex = make_float2(texcoords[idx0 * 2 + 0], texcoords[idx0 * 2 + 1]);

        emitter.area.geo.v1.pos = make_float3(p1.x, p1.y, p1.z);
        emitter.area.geo.v1.normal = make_float3(n1.x, n1.y, n1.z);
        emitter.area.geo.v1.tex = make_float2(texcoords[idx1 * 2 + 0], texcoords[idx1 * 2 + 1]);

        emitter.area.geo.v2.pos = make_float3(p2.x, p2.y, p2.z);
        emitter.area.geo.v2.normal = make_float3(n2.x, n2.y, n2.z);
        emitter.area.geo.v2.tex = make_float2(texcoords[idx2 * 2 + 0], texcoords[idx2 * 2 + 1]);

        auto v1 = emitter.area.geo.v1.pos - emitter.area.geo.v0.pos;
        auto v2 = emitter.area.geo.v2.pos - emitter.area.geo.v0.pos;
        emitter.area.area = length(cross(v1, v2)) * 0.5f;

        emitter.area.radiance = radiance;
        emitter.select_probability = select_wight * emitter.area.area;

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

std::vector<float> m_col_cdf;
std::vector<float> m_row_cdf;
std::vector<float> m_row_weight;

void BuildEnvMapCdfTable(optix::EnvMapEmitter &cu_env_map, CUdeviceptr &env_cdf_weight_cuda_memory, scene::EnvMap &env_map) noexcept {
    size_t w = env_map.radiance.bitmap.w;
    size_t h = env_map.radiance.bitmap.h;
    m_col_cdf.resize((w + 1) * h);
    m_row_cdf.resize(h + 1);
    m_row_weight.resize(h);

    size_t col_index = 0, row_index = 0;
    float row_sum = 0.f;
    m_row_cdf[row_index++] = 0.f;
    for (auto y = 0u; y < h; ++y) {
        float col_sum = 0.f;
        m_col_cdf[col_index++] = 0.f;
        for (auto x = 0u; x < w; ++x) {
            auto pixel_index = y * w + x;
            auto r = env_map.radiance.bitmap.data[pixel_index * 4 + 0];
            auto g = env_map.radiance.bitmap.data[pixel_index * 4 + 1];
            auto b = env_map.radiance.bitmap.data[pixel_index * 4 + 2];
            col_sum += optix::GetLuminance(make_float3(r, g, b));
            m_col_cdf[col_index++] = col_sum;
        }

        for (auto x = 1u; x < w; ++x)
            m_col_cdf[col_index - x - 1] /= col_sum;
        m_col_cdf[col_index - 1] = 1.f;

        float weight = std::sin((y + 0.5f) * M_PIf / h);
        m_row_weight[y] = weight;
        row_sum += col_sum * weight;
        m_row_cdf[row_index++] = row_sum;
    }

    for (auto y = 1u; y < h; ++y)
        m_row_cdf[row_index - y - 1] /= row_sum;
    m_row_cdf[row_index - 1] = 1.f;

    if (row_sum == 0)
        Pupil::Log::Warn("The environment map is completely black.");

    if (env_cdf_weight_cuda_memory) {
        Pupil::Log::Warn("Redundant environment map and the previous one will be overwritten.");
        CUDA_FREE(env_cdf_weight_cuda_memory);
    }

    cu_env_map.normalization = 1.f / (row_sum * (2.f * M_PIf / w) * (M_PIf / h));
    cu_env_map.map_size.x = w;
    cu_env_map.map_size.y = h;
}
}// namespace

namespace Pupil::optix {
EmitterHelper::EmitterHelper(scene::Scene *scene) noexcept {
    m_areas_cuda_memory = 0;
    m_points_cuda_memory = 0;
    m_directionals_cuda_memory = 0;
    m_env_cuda_memory = 0;
    m_env_cdf_weight_cuda_memory = 0;
    GenerateEmitters(scene);
}
EmitterHelper::~EmitterHelper() noexcept {
    Clear();
}

EmitterGroup EmitterHelper::GetEmitterGroup() noexcept {
    EmitterGroup ret;

    if (!m_areas_cuda_memory && m_areas.size() > 0) {
        m_areas_cuda_memory = cuda::CudaMemcpyToDevice(m_areas.data(), m_areas.size() * sizeof(Emitter));
    }
    ret.areas.SetData(m_areas_cuda_memory, m_areas.size());
    if (!m_points_cuda_memory && m_points.size() > 0) {
        m_points_cuda_memory = cuda::CudaMemcpyToDevice(m_points.data(), m_points.size() * sizeof(Emitter));
    }
    ret.points.SetData(m_points_cuda_memory, m_points.size());
    if (!m_directionals_cuda_memory && m_directionals.size() > 0) {
        m_directionals_cuda_memory = cuda::CudaMemcpyToDevice(m_directionals.data(), m_directionals.size() * sizeof(Emitter));
    }
    ret.directionals.SetData(m_directionals_cuda_memory, m_directionals.size());
    if (!m_env_cuda_memory && m_env.type != Pupil::optix::EEmitterType::None) {
        if (m_env.type == Pupil::optix::EEmitterType::EnvMap) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_env_cdf_weight_cuda_memory),
                                  sizeof(float) * (m_col_cdf.size() + m_row_cdf.size() + m_row_weight.size())));
            auto cuda_col_cdf = m_env_cdf_weight_cuda_memory;
            auto cuda_row_cdf = cuda_col_cdf + sizeof(float) * m_col_cdf.size();
            auto cuda_row_weight = cuda_row_cdf + sizeof(float) * m_row_cdf.size();

            cuda::CudaMemcpyToDevice(cuda_col_cdf, m_col_cdf.data(), sizeof(float) * m_col_cdf.size());
            cuda::CudaMemcpyToDevice(cuda_row_cdf, m_row_cdf.data(), sizeof(float) * m_row_cdf.size());
            cuda::CudaMemcpyToDevice(cuda_row_weight, m_row_weight.data(), sizeof(float) * m_row_weight.size());

            m_env.env_map.col_cdf.SetData(cuda_col_cdf, m_col_cdf.size());
            m_env.env_map.row_cdf.SetData(cuda_row_cdf, m_row_cdf.size());
            m_env.env_map.row_weight.SetData(cuda_row_weight, m_row_weight.size());
        }
        m_env_cuda_memory = cuda::CudaMemcpyToDevice(&m_env, sizeof(Emitter));
    }
    ret.env.SetData(m_env_cuda_memory);
    return ret;
}

void EmitterHelper::GenerateEmitters(scene::Scene *scene) noexcept {
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();

    // area emitters
    unsigned int area_emitter_num = 0;
    float area_select_weight_sum = 0.f;
    for (auto &&shape : scene->shapes) {
        if (!shape.is_emitter) continue;

        auto radiance = tex_mngr->GetCudaTexture(shape.emitter.area.radiance);
        float select_weight = GetWeight(shape.emitter.area.radiance);

        size_t pre_emitters_num = m_areas.size();

        switch (shape.type) {
            case scene::EShapeType::_cube: {
                area_select_weight_sum += SplitMesh(m_areas, shape.cube.vertex_num, shape.cube.face_num, shape.cube.indices,
                                                    shape.cube.positions, shape.cube.normals, shape.cube.texcoords,
                                                    shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_obj: {
                area_select_weight_sum += SplitMesh(m_areas, shape.obj.vertex_num, shape.obj.face_num, shape.obj.indices,
                                                    shape.obj.positions, shape.obj.normals, shape.obj.texcoords,
                                                    shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_rectangle: {
                area_select_weight_sum += SplitMesh(m_areas, shape.rect.vertex_num, shape.rect.face_num, shape.rect.indices,
                                                    shape.rect.positions, shape.rect.normals, shape.rect.texcoords,
                                                    shape.transform, radiance, select_weight);
            } break;
            case scene::EShapeType::_sphere: {
                Pupil::optix::Emitter emitter;
                emitter.type = Pupil::optix::EEmitterType::Sphere;
                util::Float3 o(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                util::Float3 p(o.x + shape.sphere.radius, o.y, o.z);
                o = util::Transform::TransformPoint(o, shape.transform.matrix);
                p = util::Transform::TransformPoint(p, shape.transform.matrix);

                emitter.sphere.geo.center = make_float3(o.x, o.y, o.z);
                emitter.sphere.geo.radius = length(emitter.sphere.geo.center - make_float3(p.x, p.y, p.z));
                emitter.sphere.area = 4 * 3.14159265358979323846f * emitter.sphere.geo.radius * emitter.sphere.geo.radius;
                emitter.sphere.radiance = radiance;
                emitter.select_probability = select_weight * emitter.sphere.area;

                area_select_weight_sum += emitter.select_probability;

                m_areas.emplace_back(emitter);
            } break;
        }

        shape.sub_emitters_num = static_cast<unsigned int>(
            m_areas.size() - pre_emitters_num);
        area_emitter_num += shape.sub_emitters_num;
    }

    bool emitter_valid_flag = false;
    if (area_select_weight_sum > 0.f && area_emitter_num > 0) {
        emitter_valid_flag = true;
        for (auto &&e : m_areas) {
            e.select_probability = e.select_probability / area_select_weight_sum * area_emitter_num;
        }
    }

    unsigned int other_emitter_num = 0;
    for (auto &&emitter : scene->emitters) {
        switch (emitter.type) {
            case scene::EEmitterType::ConstEnv: {
                ++other_emitter_num;
                m_env.type = Pupil::optix::EEmitterType::ConstEnv;
                m_env.const_env.color = make_float3(emitter.const_env.radiance.x,
                                                    emitter.const_env.radiance.y,
                                                    emitter.const_env.radiance.z);
                auto center = (scene->aabb.max + scene->aabb.min) * 0.5f;
                m_env.const_env.center = make_float3(center.x, center.y, center.z);
                m_env.select_probability = 1.f;
            } break;
            case scene::EEmitterType::EnvMap: {
                ++other_emitter_num;
                m_env.type = Pupil::optix::EEmitterType::EnvMap;
                m_env.env_map.radiance = tex_mngr->GetCudaTexture(emitter.env_map.radiance);
                m_env.env_map.scale = emitter.env_map.scale;
                auto center = (scene->aabb.max + scene->aabb.min) * 0.5f;
                m_env.env_map.center = make_float3(center.x, center.y, center.z);
                m_env.select_probability = 1.f;

                m_env.env_map.to_world.r0 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r0));
                m_env.env_map.to_world.r1 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r1));
                m_env.env_map.to_world.r2 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r2));

                auto to_local = emitter.env_map.transform.matrix.GetInverse();
                m_env.env_map.to_local.r0 = make_float3(ToCudaType(to_local.r0));
                m_env.env_map.to_local.r1 = make_float3(ToCudaType(to_local.r1));
                m_env.env_map.to_local.r2 = make_float3(ToCudaType(to_local.r2));

                BuildEnvMapCdfTable(m_env.env_map, m_env_cdf_weight_cuda_memory, emitter.env_map);
            } break;
                // TODO
                // case scene::EEmitterType::Point:
        }
    }

    emitter_valid_flag |= other_emitter_num > 0;

    auto emitter_num = other_emitter_num + area_emitter_num;

    for (auto &&e : m_areas) e.select_probability /= emitter_num;
    for (auto &&e : m_points) e.select_probability /= emitter_num;
    for (auto &&e : m_directionals) e.select_probability /= emitter_num;
    m_env.select_probability /= emitter_num;

    if (!emitter_valid_flag) {
        Pupil::Log::Error("No valid emitter.");
    }
}

void EmitterHelper::Reset(scene::Scene *scene) noexcept {
    Clear();
    GenerateEmitters(scene);
}

void EmitterHelper::Clear() noexcept {
    m_areas.clear();
    m_points.clear();
    m_directionals.clear();
    m_env.type = Pupil::optix::EEmitterType::None;
    CUDA_FREE(m_areas_cuda_memory);
    CUDA_FREE(m_points_cuda_memory);
    CUDA_FREE(m_directionals_cuda_memory);
    CUDA_FREE(m_env_cdf_weight_cuda_memory);
    CUDA_FREE(m_env_cuda_memory);
}

}// namespace Pupil::optix