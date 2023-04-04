#pragma once

#include "util/util.h"
#include <optix.h>

namespace Pupil::optix {
class Context : public Pupil::util::Singleton<Context> {
public:
    OptixDeviceContext context = nullptr;

    operator OptixDeviceContext() const noexcept { return context; }

    void Init() noexcept;
    void Destroy() noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

private:
    bool m_init_flag = false;
};
}// namespace Pupil::optix