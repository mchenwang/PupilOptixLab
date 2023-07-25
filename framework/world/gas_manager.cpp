#include "gas_manager.h"
#include "optix/context.h"
#include "optix/check.h"
#include "resource/shape.h"
#include "cuda/util.h"

namespace {
void CreateAccel(std::unique_ptr<Pupil::world::GAS> &gas, unsigned int vertex_num, float *vertices, unsigned int index_triplets_num, unsigned int *indices) noexcept;
void CreateAccel(std::unique_ptr<Pupil::world::GAS> &gas, float3 center, float radius) noexcept;
}// namespace

namespace {
using Mesh = std::pair<float *, uint32_t *>;

struct MeshHash {
    size_t operator()(const Mesh &mesh) const noexcept {
        size_t res = 17;
        res = res * 31 + std::hash<float *>()(mesh.first);
        res = res * 31 + std::hash<unsigned int *>()(mesh.second);
        return res;
    }
};

struct MeshCmp {
    constexpr bool operator()(const Mesh &a, const Mesh &b) const {
        return a.first == b.first && a.second == b.second;
    }
};

std::unordered_map<Mesh, OptixTraversableHandle, MeshHash, MeshCmp> m_meshs_gas;
OptixTraversableHandle m_sphere_gas = 0;
std::unordered_map<OptixTraversableHandle, CUdeviceptr> m_gas_buffers;
std::unordered_map<OptixTraversableHandle, uint32_t> m_gas_ref_cnt;
}// namespace

namespace Pupil::world {
GAS *GASManager::GetGASHandle(std::string_view shape_id) noexcept {
    auto shape_mngr = util::Singleton<resource::ShapeDataManager>::instance();
    auto shape = shape_mngr->GetShape(shape_id);
    return GetGASHandle(shape);
}

GAS *GASManager::GetGASHandle(const resource::Shape *shape) noexcept {
    if (shape == nullptr) return nullptr;

    if (m_gass.find(shape->id) != m_gass.end()) return m_gass[shape->id].get();

    auto gas = std::make_unique<GAS>();
    switch (shape->type) {
        case resource::EShapeType::_sphere: {
            if (m_sphere_gas == 0 || m_gas_buffers.find(m_sphere_gas) == m_gas_buffers.end()) {
                CreateAccel(gas, make_float3(0.f), 1.f);
                m_sphere_gas = *gas.get();
            } else {
                gas->handle = m_sphere_gas;
            }
        } break;
        case resource::EShapeType::_cube: {
            Mesh cube{ shape->cube.positions, shape->cube.indices };
            if (m_meshs_gas.find(cube) == m_meshs_gas.end() || m_gas_buffers.find(m_meshs_gas[cube]) == m_gas_buffers.end()) {
                CreateAccel(gas, shape->cube.vertex_num, shape->cube.positions, shape->cube.face_num, shape->cube.indices);
                m_meshs_gas[cube] = *gas.get();
            } else {
                gas->handle = m_meshs_gas[cube];
            }
        } break;
        case resource::EShapeType::_rectangle: {
            Mesh rect{ shape->rect.positions, shape->rect.indices };
            if (m_meshs_gas.find(rect) == m_meshs_gas.end() || m_gas_buffers.find(m_meshs_gas[rect]) == m_gas_buffers.end()) {
                CreateAccel(gas, shape->rect.vertex_num, shape->rect.positions, shape->rect.face_num, shape->rect.indices);
                m_meshs_gas[rect] = *gas.get();
            } else {
                gas->handle = m_meshs_gas[rect];
            }
        } break;
        case resource::EShapeType::_obj: {
            Mesh mesh{ shape->obj.positions, shape->obj.indices };
            if (m_meshs_gas.find(mesh) == m_meshs_gas.end() || m_gas_buffers.find(m_meshs_gas[mesh]) == m_gas_buffers.end()) {
                CreateAccel(gas, shape->obj.vertex_num, shape->obj.positions, shape->obj.face_num, shape->obj.indices);
                m_meshs_gas[mesh] = *gas.get();
            } else {
                gas->handle = m_meshs_gas[mesh];
            }
        } break;
    }

    uint32_t ref_cnt = m_gas_ref_cnt.find(gas->handle) == m_gas_ref_cnt.end() ? 0u : m_gas_ref_cnt[gas->handle];
    m_gas_ref_cnt[gas->handle] = ref_cnt + 1;
    m_gass[shape->id] = std::move(gas);
    return m_gass[shape->id].get();
}

void GASManager::Remove(std::string_view id) noexcept {
    auto it = m_gass.find(id);
    if (it == m_gass.end()) return;

    if (m_gas_ref_cnt[it->second->handle] > 0)
        m_gas_ref_cnt[it->second->handle]--;

    m_gass.erase(id.data());
}

void GASManager::ClearDanglingMemory() noexcept {
    for (auto it = m_gas_ref_cnt.begin(); it != m_gas_ref_cnt.end();) {
        if (it->second == 0) {
            if (m_gas_buffers.find(it->first) != m_gas_buffers.end())
                CUDA_FREE(m_gas_buffers[it->first]);
            m_gas_buffers.erase(it->first);

            it = m_gas_ref_cnt.erase(it);
        } else
            ++it;
    }
}

void GASManager::Destroy() noexcept {
    for (auto &&[handle, buffer] : m_gas_buffers)
        CUDA_FREE(buffer);
    m_gas_buffers.clear();
    m_gas_ref_cnt.clear();
    m_gass.clear();
    m_meshs_gas.clear();
    m_sphere_gas = 0;
}

void GAS::Create(const OptixBuildInput &input) noexcept {
    auto context = Pupil::util::Singleton<Pupil::optix::Context>::instance();
    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(*context, &accel_options, &input, 1u, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size{};
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        *context,
        0,
        &accel_options,
        &input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &handle,
        &emitProperty,// emitted property list
        1             // num emitted properties
        ));

    CUDA_FREE(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    CUdeviceptr gas_buffer;
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gas_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, handle, gas_buffer, compacted_gas_size, &handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        gas_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }

    m_gas_buffers[handle] = gas_buffer;
}
}// namespace Pupil::world

namespace {
void CreateAccel(std::unique_ptr<Pupil::world::GAS> &gas,
                 unsigned int vertex_num,
                 float *vertices,
                 unsigned int index_triplets_num,
                 unsigned int *indices) noexcept {
    const auto vertex_size = sizeof(float) * 3 * vertex_num;
    CUdeviceptr d_vertex = Pupil::cuda::CudaMemcpyToDevice(vertices, vertex_size);

    const auto index_size = sizeof(unsigned int) * 3 * index_triplets_num;
    CUdeviceptr d_index = Pupil::cuda::CudaMemcpyToDevice(indices, index_size);

    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = Pupil::cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));

    unsigned int input_flag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray = {
        .vertexBuffers = &d_vertex,
        .numVertices = vertex_num,
        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
        .vertexStrideInBytes = sizeof(float) * 3,
        .indexBuffer = d_index,
        .numIndexTriplets = index_triplets_num,
        .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
        .indexStrideInBytes = sizeof(unsigned int) * 3,
        // .preTransform = d_transform,
        .flags = &input_flag,
        .numSbtRecords = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
        .sbtIndexOffsetBuffer = d_sbt_index,
        .sbtIndexOffsetSizeInBytes = sizeof(sbt_index),
        .sbtIndexOffsetStrideInBytes = sizeof(sbt_index),
        // .transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
    };

    gas->Create(input);

    CUDA_FREE(d_vertex);
    CUDA_FREE(d_index);
    CUDA_FREE(d_sbt_index);
}

void CreateAccel(std::unique_ptr<Pupil::world::GAS> &gas,
                 float3 center, float radius) noexcept {
    CUdeviceptr d_center = Pupil::cuda::CudaMemcpyToDevice(&center, sizeof(center));
    CUdeviceptr d_radius = Pupil::cuda::CudaMemcpyToDevice(&radius, sizeof(radius));
    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = Pupil::cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));

    unsigned int input_flag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    input.sphereArray = {
        .vertexBuffers = &d_center,
        .vertexStrideInBytes = sizeof(center),
        .numVertices = 1,
        .radiusBuffers = &d_radius,
        .radiusStrideInBytes = sizeof(radius),
        .singleRadius = 1,
        .flags = &input_flag,
        .numSbtRecords = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
        .sbtIndexOffsetBuffer = d_sbt_index,
        .sbtIndexOffsetSizeInBytes = sizeof(sbt_index),
        .sbtIndexOffsetStrideInBytes = sizeof(sbt_index),
    };

    gas->Create(input);

    CUDA_FREE(d_center);
    CUDA_FREE(d_radius);
    CUDA_FREE(d_sbt_index);
}
}// namespace