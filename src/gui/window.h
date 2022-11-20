#pragma once
#include "common/util.h"

namespace gui {
class Window : public util::Singleton<Window> {
public:
    void Init();
    void Show();
    void Destroy();
};
}// namespace gui