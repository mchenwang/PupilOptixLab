#pragma once

#include "util/util.h"
#include <optix.h>

namespace Pupil::optix {
    enum class EDebugLevel {
        None,
        Minimal,
        Full
    };

    class Context : public Pupil::util::Singleton<Context> {
    public:
        operator OptixDeviceContext() const noexcept { return m_context; }

        void Init() noexcept;
        void Destroy() noexcept;
        void SetDebugLevel(EDebugLevel debug_level) noexcept { m_debug_level = debug_level; }
        auto GetDebugLevel() const noexcept { return m_debug_level; }

        bool IsInitialized() noexcept { return m_init_flag; }

    private:
        OptixDeviceContext m_context = nullptr;

        bool        m_init_flag   = false;
        EDebugLevel m_debug_level = EDebugLevel::Minimal;
    };
}// namespace Pupil::optix