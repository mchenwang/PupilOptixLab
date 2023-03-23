#include "optix/scene/scene.h"
#include "scene/scene.h"

#include "optix/context.h"
#include "optix/check.h"
#include "optix/scene/mesh.h"
#include "cuda/util.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace Pupil::optix {
Scene::Scene(const Pupil::scene::Scene *scene) noexcept {
    ResetScene(scene);
}

void Scene::ResetScene(const Pupil::scene::Scene *scene) noexcept {
    m_ros.clear();
    m_ros.reserve(scene->shapes.size());

    MeshEntity temp_mesh{};
    SphereEntity temp_sphere{};

    for (int i = 0; auto &&shape : scene->shapes) {
        switch (shape.type) {
            case scene::EShapeType::_obj:
                temp_mesh.vertex_num = shape.obj.vertex_num;
                temp_mesh.vertices = shape.obj.positions;
                temp_mesh.index_triplets_num = shape.obj.face_num;
                temp_mesh.indices = shape.obj.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_rectangle:
                temp_mesh.vertex_num = shape.rect.vertex_num;
                temp_mesh.vertices = shape.rect.positions;
                temp_mesh.index_triplets_num = shape.rect.face_num;
                temp_mesh.indices = shape.rect.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_cube:
                temp_mesh.vertex_num = shape.cube.vertex_num;
                temp_mesh.vertices = shape.cube.positions;
                temp_mesh.index_triplets_num = shape.cube.face_num;
                temp_mesh.indices = shape.cube.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh));
                break;
            case scene::EShapeType::_sphere:
                temp_sphere.center = make_float3(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                temp_sphere.radius = shape.sphere.radius;
                std::memcpy(temp_sphere.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::BuiltinSphere, &temp_sphere));
            default:
                break;
        }
    }

    CreateTopLevelAccel();
}

void Scene::CreateTopLevelAccel() noexcept {
    auto context = util::Singleton<Context>::instance();

    const auto num_instances = m_ros.size();
    std::vector<OptixInstance> instances(num_instances);

    for (auto i = 0u; i < num_instances; i++) {
        memcpy(instances[i].transform, m_ros[i]->transform, sizeof(float) * 12);
        instances[i].instanceId = i;
        instances[i].sbtOffset = i * 2;
        instances[i].visibilityMask = m_ros[i]->visibility_mask;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].traversableHandle = m_ros[i]->gas_handle;
    }

    const auto instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUdeviceptr d_instances = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void **>(d_instances),
        instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice));

    OptixBuildInput instance_input{};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        *context,
        &accel_options,
        &instance_input,
        1,// num build inputs
        &ias_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_ias_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        *context,
        0,
        &accel_options,
        &instance_input,
        1,
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        &emitProperty,
        1));

    CUDA_FREE(d_temp_buffer);
    CUDA_FREE(d_instances);

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
        CUDA_FREE(ias_buffer);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ias_buffer), compacted_ias_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, ias_handle, ias_buffer, compacted_ias_size, &ias_handle));

        CUDA_FREE(d_buffer_temp_output_ias_and_compacted_size);
    } else {
        ias_buffer = d_buffer_temp_output_ias_and_compacted_size;
    }
}

Scene::~Scene() noexcept {
    CUDA_FREE(ias_buffer);
    camera.reset();
    emitters.reset();
}
}// namespace Pupil::optix