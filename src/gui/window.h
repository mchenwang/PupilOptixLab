#pragma once
#include "common/util.h"

#include <functional>

namespace gui {
class Backend;

enum class GlobalMessage : unsigned int {
    None,
    Quit,
    Resize,
    MouseLeftButtonMove,
    MouseRightButtonMove,
    MouseWheel,
    KeyboardMove
};

class Window : public util::Singleton<Window> {
public:
    void Init() noexcept;
    GlobalMessage Show() noexcept;
    void Destroy() noexcept;

    void SetWindowMessageCallback(GlobalMessage, std::function<void()> &&) noexcept;
    void AppendGuiConsoleOperations(std::string, std::function<void()> &&) noexcept;

    void Resize(uint32_t w, uint32_t h, bool reset_window = false) noexcept;

    Backend *GetBackend() const noexcept;

    void GetWindowSize(uint32_t &w, uint32_t &h) const noexcept;

    int GetMouseLastDeltaX() const noexcept;
    int GetMouseLastDeltaY() const noexcept;
    short GetMouseWheelDelta() const noexcept;
    bool IsKeyPressed(int key) const noexcept;
};
}// namespace gui