#pragma once
#include "common/util.h"

namespace gui {
class Backend;
class Window : public util::Singleton<Window> {
public:
    void Init() noexcept;
    bool Show() noexcept;
    void Destroy() noexcept;

    void Resize(uint32_t w, uint32_t h, bool reset_window = false) noexcept;

    Backend *GetBackend() const noexcept;
};
}// namespace gui