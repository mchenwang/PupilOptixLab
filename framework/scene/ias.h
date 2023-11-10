#pragma once

#include <optix.h>
#include <vector>
#include <memory>
#include <unordered_map>

namespace Pupil {
    struct Instance;

    class IAS {
    public:
        IAS()
        noexcept;
        ~IAS() noexcept;

        operator OptixTraversableHandle() const noexcept { return m_handle; }

        void Create(std::vector<OptixInstance>& instances, unsigned int gas_offset, bool allow_update) noexcept;
        void Update(std::vector<OptixInstance>& instances) noexcept;

    private:
        OptixTraversableHandle m_handle = 0;

        CUdeviceptr m_instances_memory = 0;
        size_t      m_instances_num    = 0;

        CUdeviceptr m_ias_buffer                = 0;
        size_t*     m_compacted_ias_size_host_p = nullptr;

        CUdeviceptr m_ias_build_update_temp_buffer = 0;
        size_t      m_ias_update_temp_buffer_size  = 0;
    };

    // IASs have different instance sbt offsets for one scene.
    class IASManager {
    public:
        IASManager() noexcept;
        ~IASManager() noexcept;

        void SetInstance(const std::vector<Instance>& instances) noexcept;
        void AddInstance(const Instance& instance) noexcept;
        // TODO: edit
        // void RemoveInstance() noexcept;
        void                   UpdateInstance(const Instance* instance) noexcept;
        OptixTraversableHandle GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept;

        void Clear() noexcept;

        // void SetDirty() noexcept;
        // bool IsDirty() const noexcept;
        // void SetDirty(unsigned int gas_offset, bool allow_update) noexcept;
        // bool IsDirty(unsigned int gas_offset, bool allow_update) const noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil