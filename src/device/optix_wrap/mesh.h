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

// one to one correspondence between mesh and material
struct Mesh {
    unsigned int vertex_num;
    float *vertices; // position
    unsigned int index_triplets_num;
    unsigned int *indices;
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
    float transform[12]{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };

    RenderObject(device::Optix *, EMeshType, void *, unsigned int v_mask = 1) noexcept;
    ~RenderObject() noexcept;

    RenderObject(const RenderObject &) = delete;
    RenderObject &operator=(const RenderObject &) = delete;
};

}// namespace optix_wrap