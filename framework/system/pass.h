#pragma once

#include <string>

namespace Pupil {
namespace scene {
class Scene;
}

enum class EPassTag : uint32_t {
    None = 0,
    Pre = 1 << 0,
    Post = 1 << 1,
    Asyn = 1 << 2
};

inline static bool operator&(const EPassTag &target, const EPassTag &tag) noexcept {
    return static_cast<uint32_t>(target) & static_cast<uint32_t>(tag);
}
inline static EPassTag operator|(const EPassTag &target, const EPassTag &tag) noexcept {
    return static_cast<EPassTag>(
        static_cast<uint32_t>(target) | static_cast<uint32_t>(tag));
}

class Pass {
public:
    const std::string name;
    const EPassTag tag;
    Pass(std::string_view name, EPassTag tag = EPassTag::None) noexcept
        : name(name), tag(tag) {}

    virtual void Run() noexcept = 0;
    virtual void BeforeRunning() noexcept = 0;
    virtual void AfterRunning() noexcept = 0;
    virtual void Inspector() noexcept {}
};
}// namespace Pupil