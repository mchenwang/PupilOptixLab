#pragma once

#include "util/transform.h"
#include "util/util.h"

#include <vector>
#include <unordered_map>
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
};

struct SphereEntity {
    float3 center;
    float radius;
};

class MeshManager : util::Singleton<MeshManager> {
public:
    OptixTraversableHandle GetGASHandle(EMeshEntityType, void *) noexcept;

    void Remove(OptixTraversableHandle gas_handle) noexcept;

    void Destroy() noexcept;

private:
    std::unordered_map<OptixTraversableHandle, CUdeviceptr> m_gas_buffers;
};
}// namespace Pupil::optix