#include "emitter.h"
#include "world.h"
#include "resource/scene.h"
#include "cuda/texture.h"
#include "cuda/util.h"

#include "system/type.h"

#include <DirectXMath.h>

namespace {
using namespace Pupil;

// float SplitMesh(std::vector<Pupil::optix::Emitter> &emitters,
//                 uint32_t vertex_num, uint32_t face_num, uint32_t *indices,
//                 const float *positions, const float *normals, const float *texcoords,
//                 const util::Transform &transform, const cuda::Texture &radiance, const float select_weight) noexcept {
//     auto normal_transform = transform.matrix.GetInverse().GetTranspose();

//     float weight_sum = 0.f;

//     for (auto i = 0u; i < face_num; ++i) {
//         Pupil::optix::Emitter emitter;
//         emitter.type = Pupil::optix::EEmitterType::TriArea;

//         auto idx0 = indices[i * 3 + 0];
//         auto idx1 = indices[i * 3 + 1];
//         auto idx2 = indices[i * 3 + 2];

//         util::Float3 p0(positions[idx0 * 3 + 0], positions[idx0 * 3 + 1], positions[idx0 * 3 + 2]);
//         util::Float3 p1(positions[idx1 * 3 + 0], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2]);
//         util::Float3 p2(positions[idx2 * 3 + 0], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2]);

//         p0 = util::Transform::TransformPoint(p0, transform.matrix);
//         p1 = util::Transform::TransformPoint(p1, transform.matrix);
//         p2 = util::Transform::TransformPoint(p2, transform.matrix);

//         util::Float3 n0(normals[idx0 * 3 + 0], normals[idx0 * 3 + 1], normals[idx0 * 3 + 2]);
//         util::Float3 n1(normals[idx1 * 3 + 0], normals[idx1 * 3 + 1], normals[idx1 * 3 + 2]);
//         util::Float3 n2(normals[idx2 * 3 + 0], normals[idx2 * 3 + 1], normals[idx2 * 3 + 2]);

//         n0 = util::Transform::TransformNormal(n0, normal_transform);
//         n1 = util::Transform::TransformNormal(n1, normal_transform);
//         n2 = util::Transform::TransformNormal(n2, normal_transform);

//         emitter.area.geo.v0.pos = make_float3(p0.x, p0.y, p0.z);
//         emitter.area.geo.v0.normal = make_float3(n0.x, n0.y, n0.z);
//         emitter.area.geo.v0.tex = make_float2(texcoords[idx0 * 2 + 0], texcoords[idx0 * 2 + 1]);

//         emitter.area.geo.v1.pos = make_float3(p1.x, p1.y, p1.z);
//         emitter.area.geo.v1.normal = make_float3(n1.x, n1.y, n1.z);
//         emitter.area.geo.v1.tex = make_float2(texcoords[idx1 * 2 + 0], texcoords[idx1 * 2 + 1]);

//         emitter.area.geo.v2.pos = make_float3(p2.x, p2.y, p2.z);
//         emitter.area.geo.v2.normal = make_float3(n2.x, n2.y, n2.z);
//         emitter.area.geo.v2.tex = make_float2(texcoords[idx2 * 2 + 0], texcoords[idx2 * 2 + 1]);

//         auto v1 = emitter.area.geo.v1.pos - emitter.area.geo.v0.pos;
//         auto v2 = emitter.area.geo.v2.pos - emitter.area.geo.v0.pos;
//         emitter.area.area = length(cross(v1, v2)) * 0.5f;

//         emitter.area.radiance = radiance;
//         emitter.weight = select_weight * emitter.area.area;

//         weight_sum += emitter.weight;

//         emitters.emplace_back(emitter);
//     }

//     return weight_sum;
// }

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

void BuildEnvMapCdfTable(optix::EnvMapEmitter &cu_env_map, const resource::EnvMap &env_map) noexcept {
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

    cu_env_map.normalization = 1.f / (row_sum * (2.f * M_PIf / w) * (M_PIf / h));
    cu_env_map.map_size.x = w;
    cu_env_map.map_size.y = h;
}
}// namespace

namespace Pupil::world {
using optix::Emitter;
using optix::EmitterGroup;

EmitterHelper::EmitterHelper() noexcept {
    m_areas_cuda_memory = 0;
    m_points_cuda_memory = 0;
    m_directionals_cuda_memory = 0;
    m_env_cuda_memory = 0;
    m_env_cdf_weight_cuda_memory = 0;
    m_env.type = Pupil::optix::EEmitterType::None;
    m_dirty = true;
}
EmitterHelper::~EmitterHelper() noexcept {
    Clear();
}

void EmitterHelper::SetMeshAreaEmitter(const resource::ShapeInstance &ins, size_t offset) noexcept {
    auto normal_transform = ins.transform.matrix.GetInverse().GetTranspose();
    auto &mesh = ins.shape->mesh;

    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    auto radiance = tex_mngr->GetCudaTexture(ins.emitter.area.radiance);
    float select_weight = GetWeight(ins.emitter.area.radiance);

    for (auto i = 0u; i < ins.shape->mesh.face_num; ++i) {
        optix::Emitter emitter;
        emitter.type = optix::EEmitterType::TriArea;

        auto idx0 = mesh.indices[i * 3 + 0];
        auto idx1 = mesh.indices[i * 3 + 1];
        auto idx2 = mesh.indices[i * 3 + 2];

        util::Float3 p0(mesh.positions[idx0 * 3 + 0], mesh.positions[idx0 * 3 + 1], mesh.positions[idx0 * 3 + 2]);
        util::Float3 p1(mesh.positions[idx1 * 3 + 0], mesh.positions[idx1 * 3 + 1], mesh.positions[idx1 * 3 + 2]);
        util::Float3 p2(mesh.positions[idx2 * 3 + 0], mesh.positions[idx2 * 3 + 1], mesh.positions[idx2 * 3 + 2]);

        p0 = util::Transform::TransformPoint(p0, ins.transform.matrix);
        p1 = util::Transform::TransformPoint(p1, ins.transform.matrix);
        p2 = util::Transform::TransformPoint(p2, ins.transform.matrix);

        util::Float3 n0(mesh.normals[idx0 * 3 + 0], mesh.normals[idx0 * 3 + 1], mesh.normals[idx0 * 3 + 2]);
        util::Float3 n1(mesh.normals[idx1 * 3 + 0], mesh.normals[idx1 * 3 + 1], mesh.normals[idx1 * 3 + 2]);
        util::Float3 n2(mesh.normals[idx2 * 3 + 0], mesh.normals[idx2 * 3 + 1], mesh.normals[idx2 * 3 + 2]);

        n0 = util::Transform::TransformNormal(n0, normal_transform);
        n1 = util::Transform::TransformNormal(n1, normal_transform);
        n2 = util::Transform::TransformNormal(n2, normal_transform);

        emitter.area.geo.v0.pos = ToCudaType(p0);
        emitter.area.geo.v0.normal = ToCudaType(n0);
        emitter.area.geo.v0.tex = make_float2(mesh.texcoords[idx0 * 2 + 0], mesh.texcoords[idx0 * 2 + 1]);

        emitter.area.geo.v1.pos = ToCudaType(p1);
        emitter.area.geo.v1.normal = ToCudaType(n1);
        emitter.area.geo.v1.tex = make_float2(mesh.texcoords[idx1 * 2 + 0], mesh.texcoords[idx1 * 2 + 1]);

        emitter.area.geo.v2.pos = ToCudaType(p2);
        emitter.area.geo.v2.normal = ToCudaType(n2);
        emitter.area.geo.v2.tex = make_float2(mesh.texcoords[idx2 * 2 + 0], mesh.texcoords[idx2 * 2 + 1]);

        auto v1 = emitter.area.geo.v1.pos - emitter.area.geo.v0.pos;
        auto v2 = emitter.area.geo.v2.pos - emitter.area.geo.v0.pos;
        emitter.area.area = length(cross(v1, v2)) * 0.5f;

        emitter.area.radiance = radiance;
        emitter.weight = select_weight * emitter.area.area;

        m_areas[offset + i] = emitter;
    }
}

void EmitterHelper::SetSphereAreaEmitter(const resource::ShapeInstance &ins, size_t offset) noexcept {
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    auto radiance = tex_mngr->GetCudaTexture(ins.emitter.area.radiance);
    float select_weight = GetWeight(ins.emitter.area.radiance);

    Pupil::optix::Emitter emitter;
    emitter.type = Pupil::optix::EEmitterType::Sphere;
    util::Float3 o(ins.shape->sphere.center.x, ins.shape->sphere.center.y, ins.shape->sphere.center.z);
    util::Float3 p(o.x + ins.shape->sphere.radius, o.y, o.z);
    o = util::Transform::TransformPoint(o, ins.transform.matrix);
    p = util::Transform::TransformPoint(p, ins.transform.matrix);

    emitter.sphere.geo.center = make_float3(o.x, o.y, o.z);
    emitter.sphere.geo.radius = length(emitter.sphere.geo.center - make_float3(p.x, p.y, p.z));
    emitter.sphere.area = 4 * 3.14159265358979323846f * emitter.sphere.geo.radius * emitter.sphere.geo.radius;
    emitter.sphere.radiance = radiance;
    emitter.weight = select_weight * emitter.sphere.area;

    m_areas[offset] = emitter;
}

size_t EmitterHelper::AddAreaEmitter(const resource::ShapeInstance &ins) noexcept {
    switch (ins.shape->type) {
        case resource::EShapeType::_obj:
        case resource::EShapeType::_rectangle:
        case resource::EShapeType::_cube: {
            auto offset = m_areas.size();
            m_areas.resize(offset + ins.shape->mesh.face_num);
            SetMeshAreaEmitter(ins, offset);
        } break;
        case resource::EShapeType::_sphere: {
            auto offset = m_areas.size();
            m_areas.resize(offset + 1);
            SetSphereAreaEmitter(ins, offset);
        } break;
    }

    m_dirty = true;
    return m_areas.size();
}

void EmitterHelper::ResetAreaEmitter(const resource::ShapeInstance &ins, size_t offset) noexcept {
    switch (ins.shape->type) {
        case resource::EShapeType::_obj:
        case resource::EShapeType::_rectangle:
        case resource::EShapeType::_cube: {
            SetMeshAreaEmitter(ins, offset);
        } break;
        case resource::EShapeType::_sphere: {
            SetSphereAreaEmitter(ins, offset);
        } break;
    }

    m_dirty = true;
}

void EmitterHelper::AddEmitter(const resource::Emitter &emitter) noexcept {
    auto tex_mngr = util::Singleton<cuda::CudaTextureManager>::instance();
    switch (emitter.type) {
        case resource::EEmitterType::ConstEnv: {
            m_env.type = Pupil::optix::EEmitterType::ConstEnv;
            m_env.const_env.color = make_float3(emitter.const_env.radiance.x,
                                                emitter.const_env.radiance.y,
                                                emitter.const_env.radiance.z);
            auto aabb = util::Singleton<World>::instance()->GetAABB();
            auto center = (aabb.max + aabb.min) * 0.5f;
            m_env.const_env.center = make_float3(center.x, center.y, center.z);
            m_env.weight = 1.f;
        } break;
        case resource::EEmitterType::EnvMap: {
            m_env.type = Pupil::optix::EEmitterType::EnvMap;
            m_env.env_map.radiance = tex_mngr->GetCudaTexture(emitter.env_map.radiance);
            m_env.env_map.scale = emitter.env_map.scale;
            auto aabb = util::Singleton<World>::instance()->GetAABB();
            auto center = (aabb.max + aabb.min) * 0.5f;
            m_env.env_map.center = make_float3(center.x, center.y, center.z);
            m_env.weight = 1.f;

            m_env.env_map.to_world.r0 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r0));
            m_env.env_map.to_world.r1 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r1));
            m_env.env_map.to_world.r2 = make_float3(ToCudaType(emitter.env_map.transform.matrix.r2));

            auto to_local = emitter.env_map.transform.matrix.GetInverse();
            m_env.env_map.to_local.r0 = make_float3(ToCudaType(to_local.r0));
            m_env.env_map.to_local.r1 = make_float3(ToCudaType(to_local.r1));
            m_env.env_map.to_local.r2 = make_float3(ToCudaType(to_local.r2));

            BuildEnvMapCdfTable(m_env.env_map, emitter.env_map);
            CUDA_FREE(m_env_cdf_weight_cuda_memory);
        } break;
            // TODO
            // case resource::EEmitterType::Point:
    }

    m_dirty = true;
}

void EmitterHelper::ComputeProbability() noexcept {
    float area_weight_sum = 0.f;
    for (auto &&e : m_areas)
        area_weight_sum += e.weight;

    if (m_areas.size() > 0) {
        for (auto &&e : m_areas)
            e.select_probability = e.weight / area_weight_sum * m_areas.size();
    }

    auto emitter_num = (m_env.type == optix::EEmitterType::None ? 0 : 1) +
                       m_areas.size() + m_points.size() + m_directionals.size();
    for (auto &&e : m_areas) e.select_probability = e.select_probability / emitter_num;
    for (auto &&e : m_points) e.select_probability = e.weight / emitter_num;
    for (auto &&e : m_directionals) e.select_probability = e.weight / emitter_num;
    m_env.select_probability = m_env.weight / emitter_num;
}

EmitterGroup EmitterHelper::GetEmitterGroup() noexcept {
    EmitterGroup ret;

    if (m_dirty) {
        m_dirty = false;

        if (!m_areas_cuda_memory && m_areas.size() > 0) {
            m_areas_cuda_memory = cuda::CudaMemcpyToDevice(m_areas.data(), m_areas.size() * sizeof(Emitter));
        } else {
            cuda::CudaMemcpyToDevice(m_areas_cuda_memory, m_areas.data(), m_areas.size() * sizeof(Emitter));
        }
        if (!m_points_cuda_memory && m_points.size() > 0) {
            m_points_cuda_memory = cuda::CudaMemcpyToDevice(m_points.data(), m_points.size() * sizeof(Emitter));
        } else {
            cuda::CudaMemcpyToDevice(m_points_cuda_memory, m_points.data(), m_points.size() * sizeof(Emitter));
        }
        if (!m_directionals_cuda_memory && m_directionals.size() > 0) {
            m_directionals_cuda_memory = cuda::CudaMemcpyToDevice(m_directionals.data(), m_directionals.size() * sizeof(Emitter));
        } else {
            cuda::CudaMemcpyToDevice(m_directionals_cuda_memory, m_directionals.data(), m_directionals.size() * sizeof(Emitter));
        }
        if (m_env.type != Pupil::optix::EEmitterType::None) {
            if (!m_env_cuda_memory) {
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
            } else {
                cuda::CudaMemcpyToDevice(m_env_cuda_memory, &m_env, sizeof(Emitter));
            }
        }
    }

    ret.areas.SetData(m_areas_cuda_memory, m_areas.size());
    ret.points.SetData(m_points_cuda_memory, m_points.size());
    ret.directionals.SetData(m_directionals_cuda_memory, m_directionals.size());
    ret.env.SetData(m_env_cuda_memory);
    return ret;
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

}// namespace Pupil::world