#pragma once

namespace Pupil {
namespace scene {
class Scene;
}

class Pass {
public:
    const std::string name;
    Pass(std::string_view name) noexcept : name(name) {}

    virtual void Run() noexcept = 0;
    virtual void BeforeRunning() noexcept = 0;
    virtual void AfterRunning() noexcept = 0;

    virtual void SetScene(scene::Scene *) noexcept {}
    virtual void Inspector() noexcept {}
};
}// namespace Pupil