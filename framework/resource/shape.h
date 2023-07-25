#pragma once

#include "util/transform.h"
#include "util/enum.h"
#include "util/util.h"
#include "util/type.h"
#include "util/aabb.h"

#include "material.h"
#include "emitter.h"

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

struct ObjShape {
    bool face_normals;
    bool flip_tex_coords;
    bool flip_normals;

    uint32_t vertex_num;
    uint32_t face_num;

    float *positions;
    float *normals;
    float *texcoords;
    uint32_t *indices;
};

struct Sphere {
    bool flip_normals;
    float radius;
    util::Float3 center{};
};

struct Cube {
    bool flip_normals;
    uint32_t vertex_num;
    uint32_t face_num;

    float *positions;
    float *normals;
    float *texcoords;
    uint32_t *indices;
};

struct Rectangle {
    bool flip_normals;
    uint32_t vertex_num;
    uint32_t face_num;

    float *positions;
    float *normals;
    float *texcoords;
    uint32_t *indices;
};

struct Shape {
    EShapeType type = EShapeType::_unknown;

    std::string id;

    Material mat;
    union {
        ObjShape obj;
        Sphere sphere{};
        Cube cube;
        Rectangle rect;
    };

    bool is_emitter = false;
    Emitter emitter;

    // related to emitter generation method
    unsigned int sub_emitters_num = 0;

    util::Transform transform;
    //aabb before transformation
    util::AABB aabb{};

    Shape() noexcept {}
};

Shape *LoadShapeFromXml(const resource::xml::Object *, resource::Scene *) noexcept;

class ShapeDataManager : public util::Singleton<ShapeDataManager> {
public:
    ShapeDataManager() noexcept = default;

    void LoadShapeFromFile(std::string_view) noexcept;

    [[nodiscard]] Shape *GetShape(std::string_view) noexcept;
    [[nodiscard]] Shape *LoadObjShape(std::string_view id, std::string_view) noexcept;
    [[nodiscard]] Shape *LoadSphere(std::string_view id, float, util::Float3, bool flip_normals = false) noexcept;
    [[nodiscard]] Shape *LoadCube(std::string_view id, bool flip_normals = false) noexcept;
    [[nodiscard]] Shape *LoadRectangle(std::string_view id, bool flip_normals = false) noexcept;

    void Clear() noexcept;

private:
    struct MeshData {
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texcoords;
        std::vector<uint32_t> indices;
        util::AABB aabb{};
    };

    unsigned int m_anonymous_cnt = 0;

    std::unordered_map<std::string, std::unique_ptr<MeshData>, util::StringHash, std::equal_to<>> m_meshes;
    std::unordered_map<std::string, std::unique_ptr<Shape>, util::StringHash, std::equal_to<>> m_shapes;
};
}// namespace Pupil::resource