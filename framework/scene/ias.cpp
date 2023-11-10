#include "ias.h"
#include "gas.h"
#include "scene.h"
#include "cuda/stream.h"
#include "cuda/util.h"
#include "optix/context.h"
#include "optix/check.h"

namespace Pupil {
    IAS::IAS() noexcept {
        m_handle           = 0;
        m_instances_memory = 0;
        m_instances_num    = 0;

        m_ias_buffer = 0;

        m_ias_build_update_temp_buffer = 0;
        m_ias_update_temp_buffer_size  = 0;

        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_compacted_ias_size_host_p), sizeof(size_t), cudaHostAllocDefault));
    }

    IAS::~IAS() noexcept {
        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::None).Get();

        CUDA_FREE_ASYNC(m_instances_memory, *stream);
        CUDA_FREE_ASYNC(m_ias_buffer, *stream);
        CUDA_FREE_ASYNC(m_ias_build_update_temp_buffer, *stream);
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_compacted_ias_size_host_p)));
    }

    void IAS::Create(std::vector<OptixInstance>& instances, unsigned int gas_offset, bool allow_update) noexcept {
        for (auto i = 0u; auto&& instance : instances) {
            instance.sbtOffset = gas_offset * i;
            ++i;
        }

        auto context    = util::Singleton<optix::Context>::instance();
        m_instances_num = instances.size();

        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::IASCreation).Get();

        const auto instances_size_in_bytes = sizeof(OptixInstance) * m_instances_num;
        CUDA_FREE_ASYNC(m_instances_memory, *stream);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_instances_memory), instances_size_in_bytes, *stream));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void**>(m_instances_memory),
            instances.data(),
            instances_size_in_bytes,
            cudaMemcpyHostToDevice,
            *stream));

        OptixBuildInput instance_input{
            .type          = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            .instanceArray = {
                .instances    = m_instances_memory,
                .numInstances = static_cast<unsigned int>(m_instances_num)}};

        OptixAccelBuildOptions accel_options{};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
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

        CUDA_FREE_ASYNC(m_ias_build_update_temp_buffer, *stream);
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&m_ias_build_update_temp_buffer),
            std::max(ias_buffer_sizes.tempSizeInBytes, m_ias_update_temp_buffer_size),
            *stream));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size),
            compactedSizeOffset + 8,
            *stream));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result             = (CUdeviceptr)((char*)d_buffer_temp_output_ias_and_compacted_size + compactedSizeOffset);

        OPTIX_CHECK(optixAccelBuild(
            *context,
            *stream,
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

        CUDA_CHECK(cudaMemcpyAsync(m_compacted_ias_size_host_p, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost, *stream));
        CUDA_FREE_ASYNC(m_ias_buffer, *stream);
        stream->Synchronize();
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_ias_buffer), *m_compacted_ias_size_host_p, *stream));
        OPTIX_CHECK(optixAccelCompact(*context, *stream, m_handle, m_ias_buffer, *m_compacted_ias_size_host_p, &m_handle));
        CUDA_FREE_ASYNC(d_buffer_temp_output_ias_and_compacted_size, *stream);
    }

    void IAS::Update(std::vector<OptixInstance>& instances) noexcept {
        OptixAccelBuildOptions accel_options{
            .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE,
            .operation  = OPTIX_BUILD_OPERATION_UPDATE};

        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::IASCreation).Get();

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void**>(m_instances_memory),
            instances.data(),
            instances.size() * sizeof(OptixInstance),
            cudaMemcpyHostToDevice,
            *stream));

        OptixBuildInput instance_input{
            .type          = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            .instanceArray = {
                .instances    = m_instances_memory,
                .numInstances = static_cast<unsigned int>(m_instances_num)}};

        auto context = util::Singleton<optix::Context>::instance();
        OPTIX_CHECK(optixAccelBuild(
            *context,
            *stream,
            &accel_options,
            &instance_input,
            1,
            m_ias_build_update_temp_buffer,
            m_ias_update_temp_buffer_size,
            m_ias_buffer,
            *m_compacted_ias_size_host_p,
            &m_handle,
            nullptr,
            0));
    }

    struct IASManager::Impl {
        std::unique_ptr<IAS>                     iass[2][32]{};
        std::vector<OptixInstance>               instances;
        std::unordered_map<const Instance*, int> instance_index;

        unsigned int rebuild_flag = 0;
        unsigned int update_flag  = 0;
    };

    IASManager::IASManager() noexcept {
        m_impl = new Impl();
        Clear();
    }

    IASManager::~IASManager() noexcept {
        delete m_impl;
    }

    void IASManager::Clear() noexcept {
        for (auto i = 0u; i < 32; i++) {
            m_impl->iass[0][i] = nullptr;
            m_impl->iass[1][i] = nullptr;
        }
        m_impl->instances.clear();
        m_impl->instance_index.clear();
        m_impl->update_flag = 0u;
        m_impl->rebuild_flag |= (1ll << 32) - 1;
    }

    void IASManager::SetInstance(const std::vector<Instance>& instances) noexcept {
        m_impl->instances.clear();
        m_impl->instances.resize(instances.size());
        m_impl->instance_index.clear();

        for (auto i = 0u; i < m_impl->instances.size(); i++) {
            memcpy(m_impl->instances[i].transform, instances[i].transform.matrix.e, sizeof(float) * 12);
            m_impl->instances[i].instanceId        = i;
            m_impl->instances[i].visibilityMask    = instances[i].visibility_mask;
            m_impl->instances[i].flags             = OPTIX_INSTANCE_FLAG_NONE;
            m_impl->instances[i].traversableHandle = *instances[i].gas;

            m_impl->instance_index[&instances[i]] = i;
        }
        m_impl->rebuild_flag |= (1ll << 32) - 1;
    }

    void IASManager::AddInstance(const Instance& instance) noexcept {
        OptixInstance optix_ins;

        memcpy(optix_ins.transform, instance.transform.matrix.e, sizeof(float) * 12);
        optix_ins.instanceId        = static_cast<unsigned int>(m_impl->instances.size());
        optix_ins.visibilityMask    = instance.visibility_mask;
        optix_ins.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_ins.traversableHandle = *instance.gas;

        m_impl->instance_index[&instance] = static_cast<unsigned int>(m_impl->instances.size());

        m_impl->instances.emplace_back(optix_ins);
        m_impl->rebuild_flag |= (1ll << 32) - 1;
    }

    void IASManager::UpdateInstance(const Instance* instance) noexcept {
        if (instance == nullptr || m_impl->instance_index.find(instance) == m_impl->instance_index.end()) return;

        auto i = m_impl->instance_index[instance];
        memcpy(m_impl->instances[i].transform, instance->transform.matrix.e, sizeof(float) * 12);
        m_impl->instances[i].visibilityMask    = instance->visibility_mask;
        m_impl->instances[i].flags             = OPTIX_INSTANCE_FLAG_NONE;
        m_impl->instances[i].traversableHandle = *instance->gas;

        m_impl->update_flag |= (1ll << 32) - 1;
    }

    OptixTraversableHandle IASManager::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
        if (m_impl->iass[(int)allow_update][gas_offset] == nullptr ||
            (m_impl->rebuild_flag & (1 << gas_offset))) {
            m_impl->iass[(int)allow_update][gas_offset] = std::make_unique<IAS>();
            m_impl->iass[(int)allow_update][gas_offset]->Create(m_impl->instances, gas_offset, allow_update);
            m_impl->rebuild_flag ^= (1 << gas_offset);
        } else if (allow_update) {
            if (m_impl->update_flag & (1 << gas_offset)) {
                m_impl->iass[(int)allow_update][gas_offset]->Update(m_impl->instances);
                m_impl->update_flag ^= (1 << gas_offset);
            }
        }

        return *m_impl->iass[(int)allow_update][gas_offset];
    }

    // void IASManager::SetDirty() noexcept {
    //     m_impl->dirty_flag |= (1ll << 32) - 1;
    // }

    // bool IASManager::IsDirty() const noexcept {
    //     return m_impl->dirty_flag;
    // }

    // void IASManager::SetDirty(unsigned int gas_offset, bool allow_update) noexcept {
    //     m_impl->dirty_flag |= (1 << gas_offset);
    // }

    // bool IASManager::IsDirty(unsigned int gas_offset, bool allow_update) const noexcept {
    //     return m_impl->dirty_flag & (1 << gas_offset);
    // }

}// namespace Pupil