#pragma once

#include <optix.h>
#include <vector>
#include <memory>

namespace Pupil::optix {
struct RenderObject;

class IAS {
public:
    OptixTraversableHandle ias_handle = 0;
    IAS()
    noexcept;
    ~IAS() noexcept;

    operator OptixTraversableHandle() const noexcept { return ias_handle; }

    void Create(std::vector<OptixInstance> &instances, unsigned int gas_offset, bool allow_update) noexcept;
    void Update(std::vector<OptixInstance> &instances) noexcept;

private:
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
    OptixTraversableHandle GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept;

private:
    std::unique_ptr<IAS> m_iass[2][32]{};
    std::vector<OptixInstance> m_instances;
    unsigned int m_dirty_flag = 0;
};
}// namespace Pupil::optix