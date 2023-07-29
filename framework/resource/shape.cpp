#include "shape.h"

#include "scene.h"
#include "xml/util_loader.h"

#include "cuda/util.h"
#include "util/log.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <functional>

// static data
namespace {
// clang-format off

// XY-range [-1,1]x[-1,1]
float m_rect_positions[] = {
    -1.f, -1.f, 0.f,
     1.f, -1.f, 0.f,
     1.f,  1.f, 0.f,
    -1.f,  1.f, 0.f
};
float m_rect_normals[] = {
    0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f
};
float m_rect_texcoords[] = {
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
};
uint32_t m_rect_indices[] = { 0, 1, 2, 0, 2, 3 };

// XYZ-range [-1,-1,-1]x[1,1,1]
float m_cube_positions[] = {
    -1.f,-1.f,-1.f, -1.f,-1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f,-1.f,
     1.f,-1.f,-1.f, -1.f,-1.f,-1.f, -1.f, 1.f,-1.f,  1.f, 1.f,-1.f,
     1.f,-1.f, 1.f,  1.f,-1.f,-1.f,  1.f, 1.f,-1.f,  1.f, 1.f, 1.f,
    -1.f,-1.f, 1.f,  1.f,-1.f, 1.f,  1.f, 1.f, 1.f, -1.f, 1.f, 1.f,
    -1.f, 1.f, 1.f,  1.f, 1.f, 1.f,  1.f, 1.f,-1.f, -1.f, 1.f,-1.f,
    -1.f,-1.f,-1.f,  1.f,-1.f,-1.f,  1.f,-1.f, 1.f, -1.f,-1.f, 1.f
};
float m_cube_normals[] = {
    -1.f,0.f,0.f, -1.f,0.f,0.f, -1.f,0.f,0.f, -1.f,0.f,0.f,
    0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f, 0.f,0.f,-1.f,
    1.f,0.f,0.f, 1.f,0.f,0.f, 1.f,0.f,0.f, 1.f,0.f,0.f,
    0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f,
    0.f,1.f,0.f, 0.f,1.f,0.f, 0.f,1.f,0.f, 0.f,1.f,0.f,
    0.f,-1.f,0.f, 0.f,-1.f,0.f, 0.f,-1.f,0.f, 0.f,-1.f,0.f
};
float m_cube_texcoords[] = {
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f,
    0.f,0.f, 1.f,0.f, 1.f,1.f, 0.f,1.f
};
uint32_t m_cube_indices[] = {
    0,1,2, 0,2,3,
    4,5,6, 4,6,7,
    8,9,10, 8,10,11,
    12,13,14, 12,14,15,
    16,17,18, 16,18,19,
    20,21,22, 20,22,23
};
// clang-format on
}// namespace

namespace {
using namespace Pupil;
using namespace Pupil::resource;
using Pupil::resource::EShapeType;

template<EShapeType Tag>
struct ShapeLoader {
    ShapeInstance operator()(const xml::Object *obj, Scene *scene) {
        Pupil::Log::Warn("Unknown shape type [{}].", obj->type);
        return {};
    }
};

template<>
struct ShapeLoader<EShapeType::_cube> {
    ShapeInstance operator()(const xml::Object *obj, Scene *scene) {
        ShapeInstance ins;
        ins.name = obj->id;
        ins.shape = util::Singleton<ShapeManager>::instance()->LoadCube();
        xml::LoadBool(obj, "flip_normals", ins.flip_normals, false);
        return ins;
    }
};

template<>
struct ShapeLoader<EShapeType::_rectangle> {
    ShapeInstance operator()(const xml::Object *obj, Scene *scene) {
        ShapeInstance ins;
        ins.name = obj->id;
        ins.shape = util::Singleton<ShapeManager>::instance()->LoadRectangle();
        xml::LoadBool(obj, "flip_normals", ins.flip_normals, false);
        return ins;
    }
};

template<>
struct ShapeLoader<EShapeType::_sphere> {
    ShapeInstance operator()(const xml::Object *obj, Scene *scene) {
        util::Float3 center;
        xml::Load3Float(obj, "center", center);
        float radius;
        xml::LoadFloat(obj, "radius", radius, 1.f);

        ShapeInstance ins;
        ins.name = obj->id;
        ins.shape = util::Singleton<ShapeManager>::instance()->LoadSphere();
        xml::LoadBool(obj, "flip_normals", ins.flip_normals, false);

        util::Transform transform;
        transform.Scale(radius, radius, radius);
        transform.Translate(center.x, center.y, center.z);
        ins.transform = transform;

        return ins;
    }
};

template<>
struct ShapeLoader<EShapeType::_obj> {
    ShapeInstance operator()(const xml::Object *obj, Scene *scene) {
        auto value = obj->GetProperty("filename");
        auto path = (scene->scene_root_path / value).make_preferred();

        ShapeInstance ins;
        ins.name = obj->id;
        ins.shape = util::Singleton<ShapeManager>::instance()->LoadMeshShape(path.string());

        xml::LoadBool(obj, "face_normals", ins.face_normals, false);
        xml::LoadBool(obj, "flip_tex_coords", ins.flip_tex_coords, true);
        xml::LoadBool(obj, "flip_normals", ins.flip_normals, false);

        return ins;
    }
};

using LoaderType = std::function<ShapeInstance(const xml::Object *, Scene *)>;

#define SHAPE_LOADER(mat) ShapeLoader<EShapeType::##_##mat>()
#define SHAPE_LOADER_DEFINE(...)                             \
    const std::array<LoaderType, (size_t)EShapeType::_count> \
        S_SHAPE_LOADER = { MAP_LIST(SHAPE_LOADER, __VA_ARGS__) };

SHAPE_LOADER_DEFINE(PUPIL_SCENE_SHAPE);
}// namespace

namespace {
ShapeManager::MeshDeviceMemory m_d_cube{};
ShapeManager::MeshDeviceMemory m_d_rect{};

ShapeManager::MeshDeviceMemory GetCubeDeviceMemory() noexcept {
    if (m_d_cube.position == 0) {
        m_d_cube.position = cuda::CudaMemcpyToDevice(m_cube_positions, sizeof(m_cube_positions));
        m_d_cube.normal = cuda::CudaMemcpyToDevice(m_cube_normals, sizeof(m_cube_normals));
        m_d_cube.index = cuda::CudaMemcpyToDevice(m_cube_indices, sizeof(m_cube_indices));
        m_d_cube.texcoord = cuda::CudaMemcpyToDevice(m_cube_texcoords, sizeof(m_cube_texcoords));
    }
    return m_d_cube;
}
ShapeManager::MeshDeviceMemory GetRectDeviceMemory() noexcept {
    if (m_d_rect.position == 0) {
        m_d_rect.position = cuda::CudaMemcpyToDevice(m_rect_positions, sizeof(m_rect_positions));
        m_d_rect.normal = cuda::CudaMemcpyToDevice(m_rect_normals, sizeof(m_rect_normals));
        m_d_rect.index = cuda::CudaMemcpyToDevice(m_rect_indices, sizeof(m_rect_indices));
        m_d_rect.texcoord = cuda::CudaMemcpyToDevice(m_rect_texcoords, sizeof(m_rect_texcoords));
    }
    return m_d_rect;
}
}// namespace

namespace Pupil::resource {

ShapeInstance LoadShapeInstanceFromXml(const xml::Object *obj, Scene *scene) noexcept {
    [[unlikely]] if (obj == nullptr || scene == nullptr) {
        Pupil::Log::Warn("#LoadShapeFromXml: empty xml obj or scene obj");
        return {};
    }

    for (int i = 0; auto &&name : S_SHAPE_TYPE_NAME) {
        if (obj->type.compare(name) == 0) {
            ShapeInstance shape_ins = S_SHAPE_LOADER[i](obj, scene);

            auto bsdf_obj = obj->GetUniqueSubObject("bsdf");
            scene->LoadXmlObj(bsdf_obj, &shape_ins.mat);
            auto transform_obj = obj->GetUniqueSubObject("transform");
            util::Transform transform;
            scene->LoadXmlObj(transform_obj, &transform);
            if (shape_ins.shape->type == EShapeType::_sphere) {
                shape_ins.transform = transform.matrix * shape_ins.transform.matrix;
            } else {
                shape_ins.transform = transform;
            }

            shape_ins.is_emitter = false;
            if (auto emitter_xml_obj = obj->GetUniqueSubObject("emitter"); emitter_xml_obj) {
                scene->LoadXmlObj(emitter_xml_obj, &shape_ins.emitter);
                [[unlikely]] if (shape_ins.emitter.type != EEmitterType::Area) {
                    Pupil::Log::Warn("shape emitter not support.");
                } else
                    shape_ins.is_emitter = true;
            }
            return shape_ins;
        }
        ++i;
    }

    Pupil::Log::Warn("Unknown shape type [{}].", obj->type);
    return {};
}

void ShapeManager::LoadShapeFromFile(std::string_view file_path) noexcept {
    auto it = m_meshes.find(file_path);
    if (it != m_meshes.end()) return;

    Assimp::Importer importer;
    const auto scene = importer.ReadFile(file_path.data(), aiProcess_Triangulate);

    if (scene == nullptr) {
        Pupil::Log::Warn("Mesh load from {} failed.", file_path);
        return;
    }

    if (scene->mNumMeshes != 1) {
        Pupil::Log::Warn("Mesh load from {} failed.", file_path);
        return;
    }

    auto mesh_data = std::make_unique<MeshData>();
    uint32_t vertex_index_offset = 0;
    for (auto i = 0u; i < scene->mNumMeshes; i++) {

        const auto mesh = scene->mMeshes[i];

        bool has_normals = mesh->HasNormals();
        bool has_texcoords = mesh->HasTextureCoords(0);

        for (auto j = 0u; j < mesh->mNumVertices; j++) {
            mesh_data->positions.emplace_back(mesh->mVertices[j].x);
            mesh_data->positions.emplace_back(mesh->mVertices[j].y);
            mesh_data->positions.emplace_back(mesh->mVertices[j].z);
            mesh_data->aabb.Merge(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);

            if (has_normals) {
                mesh_data->normals.emplace_back(mesh->mNormals[j].x);
                mesh_data->normals.emplace_back(mesh->mNormals[j].y);
                mesh_data->normals.emplace_back(mesh->mNormals[j].z);
            }

            if (has_texcoords) {
                mesh_data->texcoords.emplace_back(mesh->mTextureCoords[0][j].x);
                mesh_data->texcoords.emplace_back(mesh->mTextureCoords[0][j].y);
            }
        }

        for (auto j = 0u; j < mesh->mNumFaces; j++) {
            mesh_data->indices.emplace_back(mesh->mFaces[j].mIndices[0] + vertex_index_offset);
            mesh_data->indices.emplace_back(mesh->mFaces[j].mIndices[1] + vertex_index_offset);
            mesh_data->indices.emplace_back(mesh->mFaces[j].mIndices[2] + vertex_index_offset);
        }

        vertex_index_offset += mesh->mNumVertices;
    }

    mesh_data->device_memory.position = cuda::CudaMemcpyToDevice(mesh_data->positions.data(), mesh_data->positions.size() * sizeof(float));
    mesh_data->device_memory.normal = cuda::CudaMemcpyToDevice(mesh_data->normals.data(), mesh_data->normals.size() * sizeof(float));
    mesh_data->device_memory.index = cuda::CudaMemcpyToDevice(mesh_data->indices.data(), mesh_data->indices.size() * sizeof(float));
    mesh_data->device_memory.texcoord = cuda::CudaMemcpyToDevice(mesh_data->texcoords.data(), mesh_data->texcoords.size() * sizeof(uint32_t));

    m_meshes.emplace(file_path, std::move(mesh_data));
}

ShapeManager::MeshDeviceMemory ShapeManager::GetMeshDeviceMemory(const Shape *shape) noexcept {
    if (shape->type == EShapeType::_obj) {
        if (m_meshes.find(shape->file_path) != m_meshes.end())
            return m_meshes[shape->file_path]->device_memory;
    } else if (shape->type == EShapeType::_cube) {
        return GetCubeDeviceMemory();
    } else if (shape->type == EShapeType::_rectangle) {
        return GetRectDeviceMemory();
    }
    return {};
}

Shape *ShapeManager::LoadMeshShape(std::string_view file_path) noexcept {
    if (m_meshes.find(file_path) != m_meshes.end()) {
        return m_mesh_shape[file_path.data()];
    }

    LoadShapeFromFile(file_path);
    auto it = m_meshes.find(file_path);

    auto id = m_shape_id_cnt++;
    auto shape = std::make_unique<Shape>();
    shape->id = id;
    shape->file_path = file_path;
    shape->type = EShapeType::_obj;
    shape->mesh.vertex_num = static_cast<uint32_t>(it->second->positions.size() / 3);
    shape->mesh.face_num = static_cast<uint32_t>(it->second->indices.size() / 3);
    shape->mesh.positions = it->second->positions.data();
    shape->mesh.normals = it->second->normals.size() > 0 ? it->second->normals.data() : nullptr;
    shape->mesh.texcoords = it->second->texcoords.size() > 0 ? it->second->texcoords.data() : nullptr;
    shape->mesh.indices = it->second->indices.data();
    shape->aabb = it->second->aabb;

    MeshDeviceMemory d_mesh{
        .position = it->second->device_memory.position,
        .normal = it->second->device_memory.normal,
        .index = it->second->device_memory.index,
        .texcoord = it->second->device_memory.texcoord,
    };

    m_mesh_shape[file_path.data()] = shape.get();
    m_id_shapes[id] = std::move(shape);
    return m_id_shapes[id].get();
}

Shape *ShapeManager::LoadSphere(bool flip_normals) noexcept {
    if (m_sphere) return m_sphere;

    auto id = m_shape_id_cnt++;
    auto shape = std::make_unique<Shape>();
    shape->id = id;
    shape->file_path = "sphere";
    shape->type = EShapeType::_sphere;
    shape->sphere.center = util::Float3{ 0.f };
    shape->sphere.radius = 1.f;
    shape->aabb = util::AABB{ { -1.f, -1.f, -1.f }, { 1.f, 1.f, 1.f } };

    m_sphere = shape.get();
    m_id_shapes[id] = std::move(shape);
    return m_sphere;
}

Shape *ShapeManager::LoadCube(bool flip_normals) noexcept {
    if (m_cube) return m_cube;

    auto id = m_shape_id_cnt++;
    auto shape = std::make_unique<Shape>();
    shape->id = id;
    shape->file_path = "cube";
    shape->type = EShapeType::_cube;
    shape->mesh.vertex_num = 24;
    shape->mesh.face_num = 12;
    shape->mesh.positions = m_cube_positions;
    shape->mesh.normals = m_cube_normals;
    shape->mesh.texcoords = m_cube_texcoords;
    shape->mesh.indices = m_cube_indices;
    shape->aabb = util::AABB{ { -1.f, -1.f, -1.f }, { 1.f, 1.f, 1.f } };

    m_cube = shape.get();
    m_id_shapes[id] = std::move(shape);
    return m_cube;
}

Shape *ShapeManager::LoadRectangle(bool flip_normals) noexcept {
    if (m_rect) return m_rect;

    auto id = m_shape_id_cnt++;
    auto shape = std::make_unique<Shape>();
    shape->id = id;
    shape->file_path = "rectangle";
    shape->type = EShapeType::_rectangle;
    shape->mesh.vertex_num = 4;
    shape->mesh.face_num = 2;
    shape->mesh.positions = m_rect_positions;
    shape->mesh.normals = m_rect_normals;
    shape->mesh.texcoords = m_rect_texcoords;
    shape->mesh.indices = m_rect_indices;
    shape->aabb = util::AABB{ { -1.f, -1.f, 0.f }, { 1.f, 1.f, 0.f } };

    m_rect = shape.get();
    m_id_shapes[id] = std::move(shape);
    return m_rect;
}

Shape *ShapeManager::GetShape(uint32_t id) noexcept {
    if (m_id_shapes.find(id) == m_id_shapes.end()) return nullptr;
    return m_id_shapes[id].get();
}

Shape *ShapeManager::RefShape(uint32_t id) noexcept {
    if (m_id_shapes.find(id) == m_id_shapes.end()) {
        Log::Error("RefShape failed. No shape with id [{}].", id);
        return nullptr;
    }
    size_t ref_cnt = m_shape_ref_cnt.find(id) == m_shape_ref_cnt.end() ? 0 : m_shape_ref_cnt[id];
    m_shape_ref_cnt[id] = ref_cnt + 1;
    return m_id_shapes[id].get();
}

Shape *ShapeManager::RefShape(const Shape *shape) noexcept {
    if (shape == nullptr) {
        Log::Error("Shape Ref a nullptr.");
        return nullptr;
    }
    return RefShape(shape->id);
}

void ShapeManager::Release(uint32_t id) noexcept {
    if (m_shape_ref_cnt.find(id) != m_shape_ref_cnt.end() &&
        m_shape_ref_cnt[id] > 0)
        m_shape_ref_cnt[id]--;
}

void ShapeManager::Release(const Shape *shape) noexcept {
    if (shape == nullptr) return;
    Release(shape->id);
}

void ShapeManager::Remove(uint32_t id) noexcept {
    Remove(GetShape(id));
}

void ShapeManager::Remove(const Shape *shape) noexcept {
    if (shape == nullptr) return;

    if (m_shape_ref_cnt.find(shape->id) != m_shape_ref_cnt.end() &&
        m_shape_ref_cnt[shape->id] > 0) {
        Log::Warn("Remove shape(id[{}]) without release!", shape->id);
    }
    if (shape == m_sphere)
        m_sphere = nullptr;
    else if (shape == m_cube)
        m_cube = nullptr;
    else if (shape == m_rect)
        m_rect = nullptr;
    m_mesh_shape.erase(shape->file_path);
    m_meshes.erase(shape->file_path);
    m_id_shapes.erase(shape->id);
    m_shape_ref_cnt.erase(shape->id);
}

void ShapeManager::ClearDanglingMemory() noexcept {
    for (auto it = m_shape_ref_cnt.begin(); it != m_shape_ref_cnt.end();) {
        if (it->second == 0) {
            m_mesh_shape.erase(m_id_shapes[it->first]->file_path);
            m_meshes.erase(m_id_shapes[it->first]->file_path);
            if (m_id_shapes[it->first].get() == m_sphere)
                m_sphere = nullptr;
            else if (m_id_shapes[it->first].get() == m_cube)
                m_cube = nullptr;
            else if (m_id_shapes[it->first].get() == m_rect)
                m_rect = nullptr;
            m_id_shapes.erase(it->first);
            it = m_shape_ref_cnt.erase(it);
        } else
            ++it;
    }
}

void ShapeManager::Clear() noexcept {
    m_sphere = nullptr;
    m_cube = nullptr;
    m_rect = nullptr;
    m_shape_ref_cnt.clear();
    m_id_shapes.clear();
    m_meshes.clear();
    CUDA_FREE(m_d_cube.position);
    CUDA_FREE(m_d_cube.normal);
    CUDA_FREE(m_d_cube.index);
    CUDA_FREE(m_d_cube.texcoord);
    CUDA_FREE(m_d_rect.position);
    CUDA_FREE(m_d_rect.normal);
    CUDA_FREE(m_d_rect.index);
    CUDA_FREE(m_d_rect.texcoord);
    m_shape_id_cnt = 0;
}

ShapeManager::MeshData::~MeshData() noexcept {
    CUDA_FREE(device_memory.position);
    CUDA_FREE(device_memory.normal);
    CUDA_FREE(device_memory.index);
    CUDA_FREE(device_memory.texcoord);
}
}// namespace Pupil::resource