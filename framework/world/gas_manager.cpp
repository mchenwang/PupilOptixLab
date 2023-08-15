#include "gas_manager.h"
#include "optix/context.h"
#include "optix/check.h"
#include "resource/shape.h"
#include "cuda/util.h"

#include <optix_stubs.h>

namespace Pupil::world {
GAS *GASManager::RefGAS(const resource::Shape *shape) noexcept {
    if (shape == nullptr) return nullptr;
    const uint32_t shape_id = shape->id;

    if (m_gass.find(shape_id) != m_gass.end()) {
        auto gas = m_gass[shape_id].get();
        uint32_t ref_cnt = m_gas_ref_cnt.find(gas) == m_gas_ref_cnt.end() ? 0u : m_gas_ref_cnt[gas];
        m_gas_ref_cnt[gas] = ref_cnt + 1;
        return gas;
    }

    auto gas = std::make_unique<GAS>(shape);
    gas->Create();

    m_gas_ref_cnt[gas.get()] = 1;
    m_gass[shape_id] = std::move(gas);
    return m_gass[shape_id].get();
}

void GASManager::Release(GAS *gas) noexcept {
    if (gas && m_gas_ref_cnt.find(gas) != m_gas_ref_cnt.end() &&
        m_gas_ref_cnt[gas] > 0) {
        m_gas_ref_cnt[gas]--;
    }
}

void GASManager::ClearDanglingMemory() noexcept {
    for (auto it = m_gas_ref_cnt.begin(); it != m_gas_ref_cnt.end();) {
        if (it->second == 0) {
            m_gass.erase(it->first->ref_shape->id);
            it = m_gas_ref_cnt.erase(it);
        } else
            ++it;
    }
}

void GASManager::Destroy() noexcept {
    m_gas_ref_cnt.clear();
    m_gass.clear();
}

GAS::GAS(const Pupil::resource::Shape *shape) noexcept
    : m_handle(0), ref_shape(shape), m_buffer(0) {
    util::Singleton<resource::ShapeManager>::instance()->RefShape(shape);
}

GAS::~GAS() noexcept {
    util::Singleton<resource::ShapeManager>::instance()->Release(ref_shape);
    CUDA_FREE(m_buffer);
}

void GAS::Create() noexcept {
    if (ref_shape->type == resource::EShapeType::_unknown) {
        Log::Error("GAS creation failed.");
        return;
    }
    unsigned int input_flag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};

    CUdeviceptr d_temp_mem0 = 0;
    CUdeviceptr d_temp_mem1 = 0;
    CUdeviceptr d_temp_mem2 = 0;

    if (ref_shape->type == resource::EShapeType::_sphere) {
        CUdeviceptr d_center = cuda::CudaMemcpyToDevice((void *)&ref_shape->sphere.center, sizeof(ref_shape->sphere.center));
        CUdeviceptr d_radius = cuda::CudaMemcpyToDevice((void *)&ref_shape->sphere.radius, sizeof(ref_shape->sphere.radius));
        unsigned int sbt_index = 0;
        CUdeviceptr d_sbt_index = cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));

        input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        input.sphereArray = {
            .vertexBuffers = &d_center,
            .vertexStrideInBytes = sizeof(util::Float3),
            .numVertices = 1,
            .radiusBuffers = &d_radius,
            .radiusStrideInBytes = sizeof(float),
            .singleRadius = 1,
            .flags = &input_flag,
            .numSbtRecords = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
            .sbtIndexOffsetBuffer = d_sbt_index,
            .sbtIndexOffsetSizeInBytes = sizeof(sbt_index),
            .sbtIndexOffsetStrideInBytes = sizeof(sbt_index),
        };

        d_temp_mem0 = d_center;
        d_temp_mem1 = d_radius;
        d_temp_mem2 = d_sbt_index;
    } else {
        unsigned int vertex_num = ref_shape->mesh.vertex_num;
        CUdeviceptr d_vertex = cuda::CudaMemcpyToDevice(ref_shape->mesh.positions, sizeof(float) * 3 * vertex_num);
        unsigned int index_triplets_num = ref_shape->mesh.face_num;
        CUdeviceptr d_index = cuda::CudaMemcpyToDevice(ref_shape->mesh.indices, index_triplets_num * 3 * sizeof(unsigned int));
        unsigned int sbt_index = 0;
        CUdeviceptr d_sbt_index = Pupil::cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));

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

        d_temp_mem0 = d_vertex;
        d_temp_mem1 = d_index;
        d_temp_mem2 = d_sbt_index;
    }

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
        &m_handle,
        &emitProperty,// emitted property list
        1             // num emitted properties
        ));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, m_handle, m_buffer, compacted_gas_size, &m_handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        m_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }

    CUDA_FREE(d_temp_mem0);
    CUDA_FREE(d_temp_mem1);
    CUDA_FREE(d_temp_mem2);

    CUDA_FREE(d_temp_buffer);
}
}// namespace Pupil::world