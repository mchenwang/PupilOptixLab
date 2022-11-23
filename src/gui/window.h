#pragma once
#include "common/util.h"

namespace gui {
class Backend;
class Window : public util::Singleton<Window> {
public:
    void Init() noexcept;
    bool Show() noexcept;
    void Destroy() noexcept;

    Backend *GetBackend() const noexcept;
};
}// namespace gui