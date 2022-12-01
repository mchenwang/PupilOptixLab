#pragma once

#include <vector>
#include <optix.h>
#include <vector_types.h>

namespace device {
class Optix;
}

namespace optix_wrap {
enum class EMeshType {
    Custom,
    BuiltinSphere
};

struct Vertex {
    float3 position;
    float3 normal;
};

struct TriIndex {
    unsigned int v1, v2, v3;
};

// one to one correspondence between mesh and material
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<TriIndex> indices;
    //unsigned int sbt_index;
    float transform[12];
};

struct Sphere {
    float3 center;
    float radius;
    float transform[12];
};

struct RenderObject {
    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr gas_buffer = 0;
    unsigned int visibility_mask;

    RenderObject(device::Optix *, EMeshType, void *, unsigned int v_mask = 1) noexcept;
    ~RenderObject() noexcept;
};

}// namespace optix_wrap