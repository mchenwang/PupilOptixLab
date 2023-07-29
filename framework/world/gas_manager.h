#pragma once

#include "util/util.h"

#include <optix.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace Pupil::resource {
struct Shape;
}

namespace Pupil::world {
class GAS {
public:
    const Pupil::resource::Shape *ref_shape = nullptr;

    GAS(const Pupil::resource::Shape *)
    noexcept;
    ~GAS() noexcept;

    operator OptixTraversableHandle() const noexcept { return m_handle; }

    void Create() noexcept;

private:
    OptixTraversableHandle m_handle;
    CUdeviceptr m_buffer;
};

class GASManager : util::Singleton<GASManager> {
public:
    GAS *RefGAS(const resource::Shape *) noexcept;

    void Release(GAS *) noexcept;
    void ClearDanglingMemory() noexcept;

    void Destroy() noexcept;

    // key: shape id
    std::unordered_map<uint32_t, std::unique_ptr<Pupil::world::GAS>> m_gass;
    std::unordered_map<Pupil::world::GAS *, uint32_t> m_gas_ref_cnt;
};
}// namespace Pupil::world