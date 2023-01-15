#include "shape.h"

#include "scene.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <functional>

namespace {
using scene::EShapeType;
using scene::Shape;

inline void LoadBoolParameter(const scene::xml::Object *obj, std::string_view param_name, bool &param, bool default_value = false) noexcept {
    std::string value = obj->GetProperty(param_name);
    if (value.empty())
        param = default_value;
    else {
        if (value.compare("true") == 0)
            param = true;
        else if (value.compare("false") == 0)
            param = false;
        else
            param = default_value;
    }
}

template<EShapeType Tag>
struct ShapeLoader {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        std::cout << "warring: unknown shape type [" << obj->type << "].\n";
        return {};
    }
};

template<>
struct ShapeLoader<EShapeType::_cube> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetCube();
        LoadBoolParameter(obj, "flip_normals", shape.cube.flip_normals, false);

        return shape;
    }
};

template<>
struct ShapeLoader<EShapeType::_rectangle> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetRectangle();
        LoadBoolParameter(obj, "flip_normals", shape.rect.flip_normals, false);

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
        LoadBoolParameter(obj, "flip_normals", shape.sphere.flip_normals, false);

        return shape;
    }
};

template<>
struct ShapeLoader<EShapeType::_obj> {
    Shape operator()(const scene::xml::Object *obj, scene::Scene *scene) {

        auto value = obj->GetProperty("filename");
        auto path = (scene->scene_root_path / value).make_preferred();
        util::Singleton<scene::ShapeDataManager>::instance()->LoadShapeFromFile(path.string());
        Shape shape = util::Singleton<scene::ShapeDataManager>::instance()->GetShape(path.string());

        LoadBoolParameter(obj, "face_normals", shape.obj.face_normals, false);
        LoadBoolParameter(obj, "flip_tex_coords", shape.obj.flip_tex_coords, true);
        LoadBoolParameter(obj, "flip_normals", shape.obj.flip_normals, false);

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

namespace scene {
Shape LoadShapeFromXml(const scene::xml::Object *obj, scene::Scene *scene) noexcept {
    if (obj == nullptr || scene == nullptr) {
        std::cerr << "warring: (LoadShapeFromXml) obj or scene is null.\n";
        return {};
    }

    for (int i = 0; auto &&name : S_SHAPE_TYPE_NAME) {
        if (obj->type.compare(name) == 0) {
            Shape shape = S_SHAPE_LOADER[i](obj, scene);
            auto bsdf = obj->GetUniqueSubObject("bsdf");
            scene->InvokeXmlObjLoadCallBack(bsdf, &shape.mat);
            auto transform = obj->GetUniqueSubObject("transform");
            scene->InvokeXmlObjLoadCallBack(transform, &shape.transform);
            // TODO: emitter;
            return shape;
        }
        ++i;
    }

    std::cout << "warring: unknown shape type [" << obj->type << "].\n";
    return {};
}

void ShapeDataManager::LoadShapeFromFile(std::string_view file_path) noexcept {
    Assimp::Importer importer;
    const auto scene = importer.ReadFile(file_path.data(), aiProcess_Triangulate);

    if (scene == nullptr) {
        std::cerr << "warring: Mesh load failed (" << file_path << ").\n";
        return;
    }

    if (scene->mNumMeshes != 1) {
        std::cerr << "warring: Mesh load failed (" << file_path << ").\n";
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
            shape->faces.emplace_back(mesh->mFaces[j].mIndices[0] + vertex_index_offset);
            shape->faces.emplace_back(mesh->mFaces[j].mIndices[1] + vertex_index_offset);
            shape->faces.emplace_back(mesh->mFaces[j].mIndices[2] + vertex_index_offset);
        }

        vertex_index_offset += mesh->mNumVertices;
    }

    m_shape_datas.emplace(file_path, std::move(shape));
}

Shape ShapeDataManager::GetShape(std::string_view id) noexcept {
    auto it = m_shape_datas.find(id);
    if (it == m_shape_datas.end()) {
        return GetSphere(1.f, util::float3{ 0.f, 0.f, 0.f }, false);
    }

    Shape shape;
    shape.type = EShapeType::_obj;
    shape.mat.type = material::EMatType::_unknown;
    shape.obj.face_normals = false;
    shape.obj.flip_tex_coords = true;
    shape.obj.flip_normals = false;
    shape.obj.positions = it->second->positions.data();
    shape.obj.normals = it->second->normals.size() > 0 ? it->second->normals.data() : nullptr;
    shape.obj.texcoords = it->second->texcoords.size() > 0 ? it->second->texcoords.data() : nullptr;
    shape.obj.faces = it->second->faces.data();

    return shape;
}

Shape ShapeDataManager::GetSphere(float r, util::float3 c, bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_sphere;
    shape.mat.type = material::EMatType::_unknown;
    shape.sphere.center = c;
    shape.sphere.radius = r;
    shape.sphere.flip_normals = flip_normals;

    return shape;
}

Shape ShapeDataManager::GetCube(bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_cube;
    shape.mat.type = material::EMatType::_unknown;
    shape.cube.flip_normals = flip_normals;

    return shape;
}

Shape ShapeDataManager::GetRectangle(bool flip_normals) noexcept {
    Shape shape;
    shape.type = EShapeType::_rectangle;
    shape.mat.type = material::EMatType::_unknown;
    shape.rect.flip_normals = flip_normals;

    return shape;
}

void ShapeDataManager::Clear() noexcept {
    m_shape_datas.clear();
}
}// namespace scene