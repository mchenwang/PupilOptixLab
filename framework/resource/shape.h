#pragma once

#include "util/transform.h"
#include "util/enum.h"
#include "util/util.h"
#include "util/type.h"
#include "util/aabb.h"

#include "material.h"
#include "emitter.h"

#include <cuda.h>

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <array>

namespace Pupil::resource {

class Scene;

namespace xml {
struct Object;
}

#define PUPIL_SCENE_SHAPE \
    obj, sphere, cube, rectangle

PUPIL_ENUM_DEFINE(EShapeType, PUPIL_SCENE_SHAPE)
PUPIL_ENUM_STRING_ARRAY(S_SHAPE_TYPE_NAME, PUPIL_SCENE_SHAPE)

struct Mesh {
    uint32_t vertex_num;
    uint32_t face_num;

    float *positions;
    float *normals;
    float *texcoords;
    uint32_t *indices;
};

struct Sphere {
    float radius;
    util::Float3 center{};
};

struct Shape {
    uint32_t id;
    std::string file_path;

    EShapeType type = EShapeType::_unknown;
    union {
        Mesh mesh;
        Sphere sphere{};
    };

    util::AABB aabb{};
};

struct ShapeInstance {
    std::string name = "";
    Shape *shape = nullptr;
    bool face_normals = false;
    bool flip_tex_coords = false;
    bool flip_normals = false;

    Material mat{};
    bool is_emitter = false;
    Emitter emitter{};

    util::Transform transform{};
};

ShapeInstance LoadShapeInstanceFromXml(const resource::xml::Object *, resource::Scene *) noexcept;

class ShapeManager : public util::Singleton<ShapeManager> {
public:
    struct MeshDeviceMemory {
        CUdeviceptr position = 0;
        CUdeviceptr normal = 0;
        CUdeviceptr index = 0;
        CUdeviceptr texcoord = 0;
    };

    ShapeManager() noexcept = default;

    void LoadShapeFromFile(std::string_view) noexcept;

    Shape *LoadMeshShape(std::string_view) noexcept;
    Shape *LoadSphere(bool flip_normals = false) noexcept;
    Shape *LoadCube(bool flip_normals = false) noexcept;
    Shape *LoadRectangle(bool flip_normals = false) noexcept;

    MeshDeviceMemory GetMeshDeviceMemory(const Shape *) noexcept;

    Shape *GetShape(uint32_t) noexcept;

    Shape *RefShape(uint32_t) noexcept;
    Shape *RefShape(const Shape *) noexcept;

    void Release(uint32_t) noexcept;
    void Release(const Shape *) noexcept;

    void Remove(uint32_t) noexcept;
    void Remove(const Shape *) noexcept;

    void Clear() noexcept;
    void ClearDanglingMemory() noexcept;

private:
    struct MeshData {
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texcoords;
        std::vector<uint32_t> indices;
        util::AABB aabb{};

        MeshDeviceMemory device_memory{};

        ~MeshData() noexcept;
    };

    uint32_t m_shape_id_cnt = 0;
    uint32_t m_anonymous_cnt = 0;

    // static shapes
    Shape *m_sphere = nullptr;
    Shape *m_cube = nullptr;
    Shape *m_rect = nullptr;

    // key: shape id
    std::unordered_map<uint32_t, std::unique_ptr<Shape>> m_id_shapes;
    std::unordered_map<uint32_t, size_t> m_shape_ref_cnt;
    // std::unordered_map<uint32_t, MeshDeviceMemory> m_shape_device_memory;
    // key: shape name
    // std::unordered_map<std::string, Shape *, util::StringHash, std::equal_to<>> m_name_shapes;
    // key: file path
    std::unordered_map<std::string, std::unique_ptr<MeshData>, util::StringHash, std::equal_to<>> m_meshes;
    std::unordered_map<std::string, Shape *, util::StringHash, std::equal_to<>> m_mesh_shape;
    // std::unordered_map<std::string, size_t> m_mesh_ref_cnt;
};
}// namespace Pupil::resource