#include "optix/scene/scene.h"
#include "scene/scene.h"

#include "optix/context.h"
#include "optix/check.h"
#include "optix/scene/mesh.h"
#include "cuda/util.h"

#include <optix_stubs.h>

namespace Pupil::optix {
Scene::Scene(Pupil::scene::Scene *scene, bool allow_update) noexcept {
    m_allow_update = allow_update;
    ResetScene(scene);
}

void Scene::ResetScene(Pupil::scene::Scene *scene) noexcept {
    m_ros.clear();
    m_ros.reserve(scene->shapes.size());

    MeshEntity temp_mesh{};
    SphereEntity temp_sphere{};

    for (auto &&shape : scene->shapes) {
        switch (shape.type) {
            case scene::EShapeType::_obj:
                temp_mesh.vertex_num = shape.obj.vertex_num;
                temp_mesh.vertices = shape.obj.positions;
                temp_mesh.index_triplets_num = shape.obj.face_num;
                temp_mesh.indices = shape.obj.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.id));
                break;
            case scene::EShapeType::_rectangle:
                temp_mesh.vertex_num = shape.rect.vertex_num;
                temp_mesh.vertices = shape.rect.positions;
                temp_mesh.index_triplets_num = shape.rect.face_num;
                temp_mesh.indices = shape.rect.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.id));
                break;
            case scene::EShapeType::_cube:
                temp_mesh.vertex_num = shape.cube.vertex_num;
                temp_mesh.vertices = shape.cube.positions;
                temp_mesh.index_triplets_num = shape.cube.face_num;
                temp_mesh.indices = shape.cube.indices;
                std::memcpy(temp_mesh.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.id));
                break;
            case scene::EShapeType::_sphere:
                temp_sphere.center = make_float3(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                temp_sphere.radius = shape.sphere.radius;
                std::memcpy(temp_sphere.transform, shape.transform.matrix.e, sizeof(float) * 12);
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::BuiltinSphere, &temp_sphere, shape.id));
            default:
                break;
        }
    }

    CreateTopLevelAccel();

    auto &&sensor = scene->sensor;
    camera_desc = util::CameraDesc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };

    if (!emitters)
        emitters = std::make_unique<optix::EmitterHelper>(scene);
    else
        emitters->Reset(scene);
}

void Scene::CreateTopLevelAccel() noexcept {
    auto context = util::Singleton<Context>::instance();

    m_instances.clear();
    m_instances.resize(m_ros.size());

    const auto num_instances = m_instances.size();

    for (auto i = 0u; i < num_instances; i++) {
        memcpy(m_instances[i].transform, m_ros[i]->transform.matrix.e, sizeof(float) * 12);
        m_instances[i].instanceId = i;
        m_instances[i].sbtOffset = i * 2;
        m_instances[i].visibilityMask = m_ros[i]->visibility_mask;
        m_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        m_instances[i].traversableHandle = m_ros[i]->gas_handle;

        m_ros[i]->BindScene(this, i);
    }

    const auto instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUDA_FREE(m_instances_memory);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_instances_memory), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void **>(m_instances_memory),
        m_instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice));

    OptixBuildInput instance_input{
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = m_instances_memory,
            .numInstances = static_cast<unsigned int>(num_instances) }
    };

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                               (m_allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0);
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        *context,
        &accel_options,
        &instance_input,
        1,// num build inputs
        &ias_buffer_sizes));

    m_ias_update_temp_buffer_size = ias_buffer_sizes.tempUpdateSizeInBytes;

    CUDA_FREE(m_ias_build_update_temp_buffer);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_ias_build_update_temp_buffer),
                          std::max(ias_buffer_sizes.tempSizeInBytes, m_ias_update_temp_buffer_size)));

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
        m_ias_build_update_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_ias_and_compacted_size,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        &emitProperty,
        1));

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
        CUDA_FREE(m_ias_buffer);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_ias_buffer), compacted_ias_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, ias_handle, m_ias_buffer, compacted_ias_size, &ias_handle));
        m_ias_buffer_size = compacted_ias_size;

        CUDA_FREE(d_buffer_temp_output_ias_and_compacted_size);
    } else {
        m_ias_buffer = d_buffer_temp_output_ias_and_compacted_size;
        m_ias_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }
}

OptixTraversableHandle Scene::GetIASHandle() noexcept {
    if (!m_scene_dirty || !m_allow_update) return ias_handle;
    m_scene_dirty = false;

    auto context = util::Singleton<Context>::instance();

    OptixAccelBuildOptions accel_options{
        .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE,
        .operation = OPTIX_BUILD_OPERATION_UPDATE
    };

    OptixBuildInput instance_input{
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = m_instances_memory,
            .numInstances = static_cast<unsigned int>(m_instances.size()) }
    };

    OPTIX_CHECK(optixAccelBuild(
        *context,
        0,
        &accel_options,
        &instance_input,
        1,
        m_ias_build_update_temp_buffer,
        m_ias_update_temp_buffer_size,
        m_ias_buffer,
        m_ias_buffer_size,
        &ias_handle,
        nullptr,
        0));

    CUDA_SYNC_CHECK();
    return ias_handle;
}

RenderObject *Scene::GetRenderObject(std::string_view id) const noexcept {
    for (auto &&ro : m_ros) {
        if (ro->id.compare(id) == 0)
            return ro.get();
    }

    Pupil::Log::Warn("Render Object [{}] missing.", id);
    return nullptr;
}
RenderObject *Scene::GetRenderObject(size_t index) const noexcept {
    if (index >= m_ros.size()) {
        Pupil::Log::Warn("#GetRenderObject index[{}] out of range[{}]", index, m_ros.size() - 1);
        return nullptr;
    }

    return m_ros[index].get();
}

Scene::~Scene() noexcept {
    CUDA_FREE(m_ias_buffer);
    emitters.reset();
}
}// namespace Pupil::optix