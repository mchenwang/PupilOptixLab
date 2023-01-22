#pragma once
#include "common/util.h"

namespace gui {
class Backend;

enum class GlobalMessage {
    None,
    Quit,
    Resize
};

class Window : public util::Singleton<Window> {
public:
    void Init() noexcept;
    GlobalMessage Show() noexcept;
    void Destroy() noexcept;

    void Resize(uint32_t w, uint32_t h, bool reset_window = false) noexcept;

    Backend *GetBackend() const noexcept;

    void GetWindowSize(uint32_t &w, uint32_t &h) noexcept;
};
}// namespace gui