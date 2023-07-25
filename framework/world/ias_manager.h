#pragma once

#include <optix.h>
#include <vector>
#include <memory>
#include <unordered_map>

namespace Pupil::world {
struct RenderObject;

class IAS {
public:
    IAS()
    noexcept;
    ~IAS() noexcept;

    operator OptixTraversableHandle() const noexcept { return m_handle; }

    void Create(std::vector<OptixInstance> &instances, unsigned int gas_offset, bool allow_update) noexcept;
    void Update(std::vector<OptixInstance> &instances) noexcept;

private:
    OptixTraversableHandle m_handle = 0;

    CUdeviceptr m_instances_memory = 0;
    size_t m_instances_num = 0;

    CUdeviceptr m_ias_buffer = 0;
    size_t m_ias_buffer_size = 0;

    CUdeviceptr m_ias_build_update_temp_buffer = 0;
    size_t m_ias_update_temp_buffer_size = 0;
};

// IASs have different instance sbt offsets for one scene.
class IASManager {
public:
    friend struct RenderObject;

    IASManager() noexcept;
    ~IASManager() noexcept;

    void SetInstance(const std::vector<RenderObject *> &render_objs) noexcept;
    void UpdateInstance(RenderObject *ro) noexcept;
    OptixTraversableHandle GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept;

    void SetDirty() noexcept;
    bool IsDirty() const noexcept;
    void SetDirty(unsigned int gas_offset, bool allow_update) noexcept;
    bool IsDirty(unsigned int gas_offset, bool allow_update) const noexcept;

private:
    std::unique_ptr<IAS> m_iass[2][32]{};
    std::vector<OptixInstance> m_instances;
    std::unordered_map<RenderObject *, int> m_ro_index;

    unsigned int m_dirty_flag = 0;
};
}// namespace Pupil::world