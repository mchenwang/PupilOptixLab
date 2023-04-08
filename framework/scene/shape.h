#pragma once

#include "util/transform.h"
#include "util/enum.h"
#include "util/util.h"
#include "util/type.h"
#include "util/aabb.h"

#include "material/material.h"
#include "emitter.h"

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <array>

namespace Pupil::scene {

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
    EShapeType type = EShapeType::_sphere;

    material::Material mat;
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
    util::AABB aabb{};

    Shape() noexcept {}
};

Shape LoadShapeFromXml(const scene::xml::Object *, scene::Scene *) noexcept;

class ShapeDataManager : public util::Singleton<ShapeDataManager> {
private:
    struct ShapeData {
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texcoords;
        std::vector<uint32_t> indices;
        util::AABB aabb{};
    };

    std::unordered_map<std::string, std::unique_ptr<ShapeData>, util::StringHash, std::equal_to<>> m_shape_datas;

public:
    ShapeDataManager() noexcept = default;

    void LoadShapeFromFile(std::string_view) noexcept;
    [[nodiscard]] Shape GetShape(std::string_view) noexcept;
    [[nodiscard]] Shape GetSphere(float, util::Float3, bool flip_normals = false) noexcept;
    [[nodiscard]] Shape GetCube(bool flip_normals = false) noexcept;
    [[nodiscard]] Shape GetRectangle(bool flip_normals = false) noexcept;

    void Clear() noexcept;
};
}// namespace Pupil::scene