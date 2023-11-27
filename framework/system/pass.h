#pragma once

#include <string>
#include "util/timer.h"

namespace Pupil {
    enum class EPassTag : uint32_t {
        None = 0,
        Pre  = 1 << 0,
        // Post = 1 << 1,
        // Asyn = 1 << 2,
        DisableCustomConsole = 1 << 3,
    };

    inline static bool operator&(const EPassTag& target, const EPassTag& tag) noexcept {
        return static_cast<uint32_t>(target) & static_cast<uint32_t>(tag);
    }

    inline static EPassTag operator|(const EPassTag& target, const EPassTag& tag) noexcept {
        return static_cast<EPassTag>(
            static_cast<uint32_t>(target) | static_cast<uint32_t>(tag));
    }

    class Pass {
    public:
        const std::string name;
        const EPassTag    tag;

        Pass(std::string_view name, EPassTag tag = EPassTag::None) noexcept
            : name(name), tag(tag) {}

        virtual void Run() noexcept;
        virtual void Console() noexcept;

        virtual void OnRun() noexcept       = 0;
        virtual void Synchronize() noexcept = 0;

        void Toggle() noexcept { m_enable ^= true; }
        void SetEnablility(bool enable) noexcept { m_enable = enable; }
        bool IsEnabled() const noexcept { return m_enable; }

        void AddDependentPass(Pass* pass) noexcept { m_dep_pass.push_back(pass); }

    protected:
        bool               m_enable         = true;
        double             m_last_exec_time = 0.;
        std::vector<Pass*> m_dep_pass;
    };
}// namespace Pupil