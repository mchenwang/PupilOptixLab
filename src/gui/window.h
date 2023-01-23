#pragma once
#include "common/util.h"

#include <functional>

namespace gui {
class Backend;

enum class GlobalMessage : unsigned int {
    None,
    Quit,
    Resize
};

class Window : public util::Singleton<Window> {
public:
    void Init() noexcept;
    void Show() noexcept;
    void Destroy() noexcept;

    void SetWindowMessageCallback(GlobalMessage, std::function<void()>&&) noexcept;

    void Resize(uint32_t w, uint32_t h, bool reset_window = false) noexcept;

    Backend *GetBackend() const noexcept;

    void GetWindowSize(uint32_t &w, uint32_t &h) noexcept;
};
}// namespace gui