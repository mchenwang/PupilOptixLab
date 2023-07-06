#include "shape.h"

#include "scene.h"
#include "xml/util_loader.h"

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
using namespace Pupil::scene;
using Pupil::scene::EShapeType;

template<EShapeType Tag>
struct ShapeLoader {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Pupil::Log::Warn("unknown shape type [{}].", obj->type);
        return {};
    }
};

template<>
struct ShapeLoader<EShapeType::_cube> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetCube();
        shape.id = obj->id;
        scene::xml::LoadBool(obj, "flip_normals", shape.cube.flip_normals, false);

        return shape;
    }
};

template<>
struct ShapeLoader<EShapeType::_rectangle> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetRectangle();
        shape.id = obj->id;
        scene::xml::LoadBool(obj, "flip_normals", shape.rect.flip_normals, false);

        return shape;
    }
};

template<>
struct ShapeLoader<EShapeType::_sphere> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        std::string value = obj->GetProperty("center");
        auto center = util::StrToFloat3(value);
        value = obj->GetProperty("radius");
        float radius = 1.f;
        if (!value.empty()) radius = std::stof(value);

        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetSphere(radius, center);
        shape.id = obj->id;
        scene::xml::LoadBool(obj, "flip_normals", shape.sphere.flip_normals, false);

        return shape;
    }
};

template<>
struct ShapeLoader<EShapeType::_obj> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        auto value = obj->GetProperty("filename");
        auto path = (scene->scene_root_path / value).make_preferred();
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetShape(path.string());
        shape.id = obj->id;

        scene::xml::LoadBool(obj, "face_normals", shape.obj.face_normals, false);
        scene::xml::LoadBool(obj, "flip_tex_coords", shape.obj.flip_tex_coords, true);
        scene::xml::LoadBool(obj, "flip_normals", shape.obj.flip_normals, false);

        return shape;
    }
};

using LoaderType = std::function<Shape(const scene::xml::Object *, scene::Scene *)>;

#define SHAPE_LOADER(mat) ShapeLoader<EShapeType::##_##mat>()
#define SHAPE_LOADER_DEFINE(...)                             \
    const std::array<LoaderType, (size_t)EShapeType::_count> \
        S_SHAPE_LOADER = { MAP_LIST(SHAPE_LOADER, __VA_ARGS__) };

SHAPE_LOADER_DEFINE(PUPIL_SCENE_SHAPE);
}// namespace

namespace Pupil::scene {
Shape LoadShapeFromXml(const scene::xml::Object *obj, scene::Scene *scene) noexcept {
    if (obj == nullptr || scene == nullptr) {
        Pupil::Log::Warn("obj or scene is null.\n\tlocation: LoadShapeFromXml().");
        return {};
    }

    for (int i = 0; auto &&name : S_SHAPE_TYPE_NAME) {
        if (obj->type.compare(name) == 0) {
            Shape shape = S_SHAPE_LOADER[i](obj, scene);
            auto bsdf = obj->GetUniqueSubObject("bsdf");
            scene->InvokeXmlObjLoadCallBack(bsdf, &shape.mat);
            auto transform = obj->GetUniqueSubObject("transform");
            scene->InvokeXmlObjLoadCallBack(transform, &shape.transform);
            shape.aabb.Transform(shape.transform);

            shape.is_emitter = false;
            if (auto emitter_xml_obj = obj->GetUniqueSubObject("emitter"); emitter_xml_obj) {
                scene->InvokeXmlObjLoadCallBack(emitter_xml_obj, &shape.emitter);
                if (shape.emitter.type != EEmitterType::Area) {
                    Pupil::Log::Warn("shape emitter not support.");
                } else
                    shape.is_emitter = true;
            }
            return shape;
        }
        ++i;
    }

    Pupil::Log::Warn("unknown shape type [{}].", obj->type);
    return {};
}

void ShapeDataManager::LoadShapeFromFile(std::string_view file_path) noexcept {
    auto it = m_shape_datas.find(file_path);
    if (it != m_shape_datas.end()) return;

    Assimp::Importer importer;
    const auto scene = importer.ReadFile(file_path.data(), aiProcess_Triangulate);

    if (scene == nullptr) {
        Pupil::Log::Warn("Mesh load failed.\n\tlocation: {}.", file_path);
        return;
    }

    if (scene->mNumMeshes != 1) {
        Pupil::Log::Warn("Mesh load failed.\n\tlocation: {}.", file_path);
        return;
    }

    auto shape = std::make_unique<ShapeData>();
    uint32_t vertex_index_offset = 0;
    for (auto i = 0u; i < scene->mNumMeshes; i++) {

        const auto mesh = scene->mMeshes[i];

        bool has_normals = mesh->HasNormals();
        bool has_texcoords = mesh->HasTextureCoords(0);

        for (auto j = 0u; j < mesh->mNumVertices; j++) {
            shape->positions.emplace_back(mesh->mVertices[j].x);
            shape->positions.emplace_back(mesh->mVertices[j].y);
            shape->positions.emplace_back(mesh->mVertices[j].z);
            shape->aabb.Merge(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);

            if (has_normals) {
                shape->normals.emplace_back(mesh->mNormals[j].x);
                shape->normals.emplace_back(mesh->mNormals[j].y);
                shape->normals.emplace_back(mesh->mNormals[j].z);
            }

            if (has_texcoords) {
                shape->texcoords.emplace_back(mesh->mTextureCoords[0][j].x);
                shape->texcoords.emplace_back(mesh->mTextureCoords[0][j].y);
            }
        }

        for (auto j = 0u; j < mesh->mNumFaces; j++) {
            shape->indices.emplace_back(mesh->mFaces[j].mIndices[0] + vertex_index_offset);
            shape->indices.emplace_back(mesh->mFaces[j].mIndices[1] + vertex_index_offset);
            shape->indices.emplace_back(mesh->mFaces[j].mIndices[2] + vertex_index_offset);
        }

        vertex_index_offset += mesh->mNumVertices;
    }

    m_shape_datas.emplace(file_path, std::move(shape));
}

Shape ShapeDataManager::GetShape(std::string_view id) noexcept {
    auto it = m_shape_datas.find(id);
    if (it == m_shape_datas.end()) {
        this->LoadShapeFromFile(id);
        it = m_shape_datas.find(id);
        [[unlikely]] if (it == m_shape_datas.end()) {
            return GetSphere(1.f, util::Float3{ 0.f, 0.f, 0.f }, false);
        }
    }

    Shape shape;
    shape.type = EShapeType::_obj;
    shape.mat.type = material::EMatType::Unknown;
    shape.obj.face_normals = false;
    shape.obj.flip_tex_coords = true;
    shape.obj.flip_normals = false;
    shape.obj.vertex_num = static_cast<uint32_t>(it->second->positions.size() / 3);
    shape.obj.face_num = static_cast<uint32_t>(it->second->indices.size() / 3);
    shape.obj.positions = it->second->positions.data();
    shape.obj.normals = it->second->normals.size() > 0 ? it->second->normals.data() : nullptr;
    shape.obj.texcoords = it->second->texcoords.size() > 0 ? it->second->texcoords.data() : nullptr;
    shape.obj.indices = it->second->indices.data();
    shape.aabb = it->second->aabb;

    return shape;
}

Shape ShapeDataManager::GetSphere(float r, util::Float3 c, bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_sphere;
    shape.mat.type = material::EMatType::Unknown;
    shape.sphere.center = c;
    shape.sphere.radius = r;
    shape.sphere.flip_normals = flip_normals;
    shape.aabb = util::AABB{
        { c.x - r, c.y - r, c.z - r },
        { c.x + r, c.y + r, c.z + r }
    };

    return shape;
}

Shape ShapeDataManager::GetCube(bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_cube;
    shape.mat.type = material::EMatType::Unknown;
    shape.cube.flip_normals = flip_normals;
    shape.cube.vertex_num = 24;
    shape.cube.face_num = 12;
    shape.cube.positions = m_cube_positions;
    shape.cube.normals = m_cube_normals;
    shape.cube.texcoords = m_cube_texcoords;
    shape.cube.indices = m_cube_indices;
    shape.aabb = util::AABB{ { -1.f, -1.f, -1.f }, { 1.f, 1.f, 1.f } };

    return shape;
}

Shape ShapeDataManager::GetRectangle(bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_rectangle;
    shape.mat.type = material::EMatType::Unknown;
    shape.rect.flip_normals = flip_normals;
    shape.rect.vertex_num = 4;
    shape.rect.face_num = 2;
    shape.rect.positions = m_rect_positions;
    shape.rect.normals = m_rect_normals;
    shape.rect.texcoords = m_rect_texcoords;
    shape.rect.indices = m_rect_indices;
    shape.aabb = util::AABB{ { -1.f, -1.f, 0.f }, { 1.f, 1.f, 0.f } };

    return shape;
}

void ShapeDataManager::Clear() noexcept {
    m_shape_datas.clear();
}
}// namespace Pupil::scene