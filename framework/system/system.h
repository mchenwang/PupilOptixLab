#pragma once

#include "util/util.h"
#include "pass.h"

#include <filesystem>
#include <memory>

namespace Pupil {
    namespace Event {
        // dispatcher
        constexpr const char* DispatcherMain   = "Dispatcher Main";
        constexpr const char* DispatcherRender = "Dispatcher Render";

        // system event
        constexpr const char* FrameDone        = "Frame Done";
        constexpr const char* RenderPause      = "Render Pause";
        constexpr const char* RenderContinue   = "Render Continue";
        constexpr const char* RenderRestart    = "Render Restart";
        constexpr const char* RequestSceneLoad = "Request Scene Load";
        constexpr const char* SceneLoading     = "Scene Loading";
        constexpr const char* RequestQuit      = "Request Quit";
        constexpr const char* LimitRenderRate  = "Limit Render Rate";

    }// namespace Event

    class System : public util::Singleton<System> {
    public:
        System() noexcept;
        ~System() noexcept;

        void Init(bool has_window = true) noexcept;
        void Run() noexcept;
        void Destroy() noexcept;

        void SetFrameRateLimit(int limit) noexcept;

        void AddPass(Pass*) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil