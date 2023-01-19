#include "mesh.h"
#include "../optix_device.h"
#include "../error_handle.h"

#include "cuda_util/util.h"

using namespace optix_wrap;

namespace {
// the mesh just has one material, so the sbt_index_offset must be 0
void CreateAccel(device::Optix *device, Mesh *mesh, RenderObject *ro) {
    const auto vertex_size = sizeof(float) * 3 * mesh->vertex_num;
    CUdeviceptr d_vertex = cuda::CudaMemcpy(mesh->vertices, vertex_size);

    const auto index_size = sizeof(unsigned int) * 3 * mesh->index_triplets_num;
    CUdeviceptr d_index = cuda::CudaMemcpy(mesh->indices, index_size);

    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = cuda::CudaMemcpy(&sbt_index, sizeof(sbt_index));
    CUdeviceptr d_transform = cuda::CudaMemcpy(mesh->transform, sizeof(float) * 12);

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
        .preTransform = d_transform,
        .flags = &input_flag,
        .numSbtRecords = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
        .sbtIndexOffsetBuffer = d_sbt_index,
        .sbtIndexOffsetSizeInBytes = sizeof(sbt_index),
        .sbtIndexOffsetStrideInBytes = sizeof(sbt_index),
        .transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
    };

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(device->context, &accel_options, &input, 1u, &gas_buffer_sizes));

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
        device->context,
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
    CUDA_FREE(d_transform);
    CUDA_FREE(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ro->gas_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(device->context, 0, ro->gas_handle, ro->gas_buffer, compacted_gas_size, &ro->gas_handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        ro->gas_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
void CreateAccel(device::Optix *device, Sphere *sphere, RenderObject *ro) {
    CUdeviceptr d_center = cuda::CudaMemcpy(&sphere->center, sizeof(sphere->center));
    CUdeviceptr d_radius = cuda::CudaMemcpy(&sphere->radius, sizeof(sphere->radius));
    unsigned int sbt_index = 0;
    CUdeviceptr d_sbt_index = cuda::CudaMemcpy(&sbt_index, sizeof(sbt_index));
    CUdeviceptr d_transform = cuda::CudaMemcpy(sphere->transform, sizeof(float) * 12);

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
    OPTIX_CHECK(optixAccelComputeMemoryUsage(device->context, &accel_options, &input, 1u, &gas_buffer_sizes));

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
        device->context,
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
    CUDA_FREE(d_transform);
    CUDA_FREE(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ro->gas_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(device->context, 0, ro->gas_handle, ro->gas_buffer, compacted_gas_size, &ro->gas_handle));

        CUDA_FREE(d_buffer_temp_output_gas_and_compacted_size);
    } else {
        ro->gas_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
}// namespace

RenderObject::RenderObject(device::Optix *device, EMeshType type, void *mesh, unsigned int v_mask) noexcept
    : gas_handle(0), gas_buffer(0), visibility_mask(v_mask), transform() {
    switch (type) {
        case optix_wrap::EMeshType::Custom:
            CreateAccel(device, (Mesh *)mesh, this);
            break;
        case optix_wrap::EMeshType::BuiltinSphere:
            CreateAccel(device, (Sphere *)mesh, this);
            break;
        default:
            break;
    }
}

RenderObject::~RenderObject() noexcept {
    CUDA_FREE(gas_buffer);
}
