#pragma once

#include "util/transform.h"

#include <vector>
#include <optix.h>
#include <vector_types.h>

namespace Pupil::optix {
enum class EMeshEntityType {
    Custom,
    BuiltinSphere
};

// one to one correspondence between mesh and material
struct MeshEntity {
    unsigned int vertex_num;
    float *vertices;// position
    unsigned int index_triplets_num;
    unsigned int *indices;
    float transform[12];
};

struct SphereEntity {
    float3 center;
    float radius;
    float transform[12];
};

class Scene;

struct RenderObject {
    std::string id;
    Scene *scene = nullptr;
    int instance_index = -1;

    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr gas_buffer = 0;
    unsigned int visibility_mask = 1;
    util::Transform transform{};

    RenderObject(EMeshEntityType, void *, std::string_view id = "", unsigned int v_mask = 1) noexcept;
    ~RenderObject() noexcept;

    RenderObject(const RenderObject &) = delete;
    RenderObject &operator=(const RenderObject &) = delete;

    void BindScene(Scene *, int) noexcept;

    void UpdateTransform(const util::Transform &new_transform) noexcept;
    void ApplyTransform(const util::Transform &new_transform) noexcept;
};

}// namespace Pupil::optix