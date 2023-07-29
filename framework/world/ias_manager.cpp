#include "ias_manager.h"
#include "render_object.h"

#include "optix/context.h"
#include "optix/check.h"
#include "cuda/util.h"

#include <optix_stubs.h>

namespace Pupil::world {
IAS::IAS() noexcept {
    m_handle = 0;
    m_instances_memory = 0;
    m_instances_num = 0;

    m_ias_buffer = 0;
    m_ias_buffer_size = 0;

    m_ias_build_update_temp_buffer = 0;
    m_ias_update_temp_buffer_size = 0;
}

IAS::~IAS() noexcept {
    CUDA_FREE(m_instances_memory);
    CUDA_FREE(m_ias_buffer);
    CUDA_FREE(m_ias_build_update_temp_buffer);
}

void IAS::Create(std::vector<OptixInstance> &instances, unsigned int gas_offset, bool allow_update) noexcept {
    for (auto i = 0u; auto &&instance : instances) {
        instance.sbtOffset = gas_offset * i;
        ++i;
    }

    auto context = util::Singleton<optix::Context>::instance();
    m_instances_num = instances.size();

    const auto instances_size_in_bytes = sizeof(OptixInstance) * m_instances_num;
    CUDA_FREE(m_instances_memory);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_instances_memory), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void **>(m_instances_memory),
        instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice));

    OptixBuildInput instance_input{
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = m_instances_memory,
            .numInstances = static_cast<unsigned int>(m_instances_num) }
    };

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                               (allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0);
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
        &m_handle,
        &emitProperty,
        1));

    size_t compacted_ias_size;
    CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes) {
        CUDA_FREE(m_ias_buffer);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_ias_buffer), compacted_ias_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(*context, 0, m_handle, m_ias_buffer, compacted_ias_size, &m_handle));
        m_ias_buffer_size = compacted_ias_size;

        CUDA_FREE(d_buffer_temp_output_ias_and_compacted_size);
    } else {
        m_ias_buffer = d_buffer_temp_output_ias_and_compacted_size;
        m_ias_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    }
}

void IAS::Update(std::vector<OptixInstance> &instances) noexcept {
    OptixAccelBuildOptions accel_options{
        .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE,
        .operation = OPTIX_BUILD_OPERATION_UPDATE
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void **>(m_instances_memory),
        instances.data(),
        instances.size() * sizeof(OptixInstance),
        cudaMemcpyHostToDevice));

    OptixBuildInput instance_input{
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = m_instances_memory,
            .numInstances = static_cast<unsigned int>(m_instances_num) }
    };

    auto context = util::Singleton<optix::Context>::instance();
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
        &m_handle,
        nullptr,
        0));

    CUDA_SYNC_CHECK();
}

IASManager::IASManager() noexcept {
    for (auto i = 0u; i < 32; i++) {
        m_iass[0][i] = nullptr;
        m_iass[1][i] = nullptr;
    }
    m_dirty_flag = 0u;
}

IASManager::~IASManager() noexcept {
    m_dirty_flag = 0u;
}

void IASManager::SetInstance(const std::vector<RenderObject *> &render_objs) noexcept {
    m_instances.clear();
    m_instances.resize(render_objs.size());
    m_ro_index.clear();

    for (auto i = 0u; i < m_instances.size(); i++) {
        memcpy(m_instances[i].transform, render_objs[i]->transform.matrix.e, sizeof(float) * 12);
        m_instances[i].instanceId = i;
        // m_instances[i].sbtOffset = i;
        m_instances[i].visibilityMask = render_objs[i]->visibility_mask;
        m_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        m_instances[i].traversableHandle = *render_objs[i]->gas;

        m_ro_index[render_objs[i]] = i;
    }

    for (auto i = 0u; i < 32; i++) {
        m_iass[0][i].reset();
        m_iass[1][i].reset();
    }
}

void IASManager::UpdateInstance(RenderObject *ro) noexcept {
    if (ro == nullptr || m_ro_index.find(ro) == m_ro_index.end()) return;

    auto i = m_ro_index[ro];
    memcpy(m_instances[i].transform, ro->transform.matrix.e, sizeof(float) * 12);
    m_instances[i].visibilityMask = ro->visibility_mask;
    m_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
    m_instances[i].traversableHandle = *ro->gas;

    SetDirty();
}

OptixTraversableHandle IASManager::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
    if (m_iass[(int)allow_update][gas_offset] == nullptr) {
        m_iass[(int)allow_update][gas_offset] = std::make_unique<IAS>();
        m_iass[(int)allow_update][gas_offset]->Create(m_instances, gas_offset, allow_update);
    } else if (allow_update) {
        if (m_dirty_flag & (1 << gas_offset)) {
            m_iass[(int)allow_update][gas_offset]->Update(m_instances);
            m_dirty_flag ^= (1 << gas_offset);
        }
    }

    return *m_iass[(int)allow_update][gas_offset];
}

void IASManager::SetDirty() noexcept {
    m_dirty_flag |= (1ll << 32) - 1;
}

bool IASManager::IsDirty() const noexcept {
    return m_dirty_flag;
}

void IASManager::SetDirty(unsigned int gas_offset, bool allow_update) noexcept {
    m_dirty_flag |= (1 << gas_offset);
}

bool IASManager::IsDirty(unsigned int gas_offset, bool allow_update) const noexcept {
    return m_dirty_flag & (1 << gas_offset);
}
}// namespace Pupil::world