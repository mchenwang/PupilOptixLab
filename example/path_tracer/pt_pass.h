#pragma once

#include "system/pass.h"
#include "optix/pass.h"

namespace Pupil::pt {
    class PTPass : public Pass, public optix::Pass {
    public:
        PTPass(std::string_view name = "Path Tracing") noexcept;
        ~PTPass() noexcept;
        virtual void OnRun() noexcept override;
        virtual void Console() noexcept override;

    private:
        void InitPipeline() noexcept;
        void BindingEventCallback() noexcept;

        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::pt