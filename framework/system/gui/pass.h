#pragma once

#include "util/util.h"
#include "system/pass.h"

#include <functional>

namespace Pupil::Gui {
    namespace Event {
        constexpr const char* WindowMinimized           = "Window Minimized";
        constexpr const char* WindowResized             = "Window Resized";
        constexpr const char* CanvasDisplayTargetChange = "Canvas Display Target Change";
    }// namespace Event

    class Pass final : public Pupil::Pass, public util::Singleton<Gui::Pass> {
    public:
        Pass() noexcept;
        ~Pass() noexcept;

        void Init() noexcept;
        void Destroy() noexcept;

        virtual void OnRun() noexcept override;

        using CustomConsole = std::function<void()>;
        void RegisterConsole(std::string_view, CustomConsole&&) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::Gui