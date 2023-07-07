#include "mesh.h"
#include "ias_manager.h"
#include "optix/context.h"
#include "optix/check.h"
#include "optix/scene/scene.h"

#include "cuda/util.h"

#include <optix_stubs.h>

using namespace Pupil::optix;

namespace {
// the mesh just has one material, so the sbt_index_offset must be 0
void CreateAccel(Context *context, MeshEntity *mesh, RenderObject *ro) {
    const auto vertex_size = sizeof(float) * 3 * mesh->vertex_num;
    CUdeviceptr d_vertex = Pupil::cuda::CudaMemcpyToDevice(mesh->vertices, vertex_size);

    const auto index_size = sizeof(unsigned int) * 3 * mesh->index_triplets_num;
    CUdeviceptr d_index = Pupil::cuda::CudaMemcpyToDevice(mesh->indices, index_size);

    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = Pupil::cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));
    // CUdeviceptr d_transform = cuda::CudaMemcpyToDevice(mesh->transform, sizeof(float) * 12);

    unsigned int input_flag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray = {
        .vertexBuffers = &d_vertex,
        .numVertices = mesh->vertex_num,
        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
        .vertexStrideInBytes = sizeof(float) * 3,
        .indexBuffer = d_index,
        .numIndexTriplets = mesh->index_triplets_num,
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
        &ro->gas_handle,
        &emitProperty,// emitted property list
        1             // num emitted properties
        ));

    CUDA_FREE(d_vertex);
    CUDA_FREE(d_index);
    CUDA_FREE(d_sbt_index);
    // CUDA_FREE(d_transform);
    CUDA_FREE(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ro->gas_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, ro->gas_handle, ro->gas_buffer, compacted_gas_size, &ro->gas_handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        ro->gas_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
void CreateAccel(Context *context, SphereEntity *sphere, RenderObject *ro) {
    CUdeviceptr d_center = Pupil::cuda::CudaMemcpyToDevice(&sphere->center, sizeof(sphere->center));
    CUdeviceptr d_radius = Pupil::cuda::CudaMemcpyToDevice(&sphere->radius, sizeof(sphere->radius));
    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = Pupil::cuda::CudaMemcpyToDevice(&sbt_index, sizeof(sbt_index));
    // CUdeviceptr d_transform = cuda::CudaMemcpyToDevice(sphere->transform, sizeof(float) * 12);

    unsigned int input_flag = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    input.sphereArray = {
        .vertexBuffers = &d_center,
        .vertexStrideInBytes = sizeof(sphere->center),
        .numVertices = 1,
        .radiusBuffers = &d_radius,
        .radiusStrideInBytes = sizeof(sphere->radius),
        .singleRadius = 1,
        .flags = &input_flag,
        .numSbtRecords = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
        .sbtIndexOffsetBuffer = d_sbt_index,
        .sbtIndexOffsetSizeInBytes = sizeof(sbt_index),
        .sbtIndexOffsetStrideInBytes = sizeof(sbt_index),
    };

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
        &ro->gas_handle,
        &emitProperty,// emitted property list
        1             // num emitted properties
        ));

    CUDA_FREE(d_center);
    CUDA_FREE(d_radius);
    CUDA_FREE(d_sbt_index);
    // CUDA_FREE(d_transform);
    CUDA_FREE(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ro->gas_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, ro->gas_handle, ro->gas_buffer, compacted_gas_size, &ro->gas_handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        ro->gas_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
}// namespace

RenderObject::RenderObject(EMeshEntityType type, void *mesh, std::string_view id, unsigned int v_mask) noexcept
    : id(id), gas_handle(0), gas_buffer(0), visibility_mask(v_mask), transform() {
    auto context = util::Singleton<Context>::instance();

    switch (type) {
        case Pupil::optix::EMeshEntityType::Custom: {
            auto m = static_cast<MeshEntity *>(mesh);
            CreateAccel(context, m, this);
            std::memcpy(transform.matrix.e, m->transform, 12 * sizeof(float));
        } break;
        case Pupil::optix::EMeshEntityType::BuiltinSphere: {
            auto m = static_cast<SphereEntity *>(mesh);
            CreateAccel(context, m, this);
            std::memcpy(transform.matrix.e, m->transform, 12 * sizeof(float));
        } break;
        default:
            break;
    }
}

void RenderObject::BindScene(Scene *scene, int instance_index) noexcept {
    if (scene == nullptr) {
        Log::Error("RenderObject bind with null scene.");
        return;
    }

    if (instance_index < 0 ||
        (scene->m_ias_manager &&
         instance_index >= scene->m_ias_manager->m_instances.size())) {
        Log::Error("RenderObject instance index[{}] out of range[0, {}].",
                   instance_index, scene->m_ias_manager->m_instances.size() - 1);
        return;
    }

    this->scene = scene;
    this->instance_index = instance_index;
}

void RenderObject::UpdateTransform(const util::Transform &new_transform) noexcept {
    transform = new_transform;
    if (scene && instance_index != -1 && scene->m_ias_manager) {
        auto &instances = scene->m_ias_manager->m_instances;
        memcpy(instances[instance_index].transform, transform.matrix.e, sizeof(float) * 12);

        scene->m_ias_manager->m_dirty_flag |= (1ll << 32) - 1;
    }
}
void RenderObject::ApplyTransform(const util::Transform &new_transform) noexcept {
    transform.matrix = new_transform.matrix * transform.matrix;
    if (scene && instance_index != -1) {
        auto &instances = scene->m_ias_manager->m_instances;
        memcpy(instances[instance_index].transform, transform.matrix.e, sizeof(float) * 12);

        scene->m_ias_manager->m_dirty_flag |= (1ll << 32) - 1;
    }
}

RenderObject::~RenderObject() noexcept {
    CUDA_FREE(gas_buffer);
    gas_buffer = 0;
}
