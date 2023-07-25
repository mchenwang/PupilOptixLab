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
    GAS()
    noexcept = default;
    ~GAS() noexcept = default;

    operator OptixTraversableHandle() const noexcept { return handle; }

    void Create(const OptixBuildInput &input) noexcept;

    OptixTraversableHandle handle = 0;
};

class GASManager : util::Singleton<GASManager> {
public:
    GAS *GetGASHandle(std::string_view) noexcept;
    GAS *GetGASHandle(const resource::Shape *) noexcept;

    void Remove(std::string_view) noexcept;
    void ClearDanglingMemory() noexcept;

    void Destroy() noexcept;

private:
    std::unordered_map<std::string, std::unique_ptr<GAS>, util::StringHash, std::equal_to<>> m_gass;
};
}// namespace Pupil::world