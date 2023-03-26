#pragma once

namespace Pupil {
class Pass {
public:
    const std::string pass_name;
    Pass(std::string_view name) noexcept : pass_name(name) {}

    virtual void Run() noexcept = 0;
    virtual void BeforeRunning() noexcept = 0;
    virtual void AfterRunning() noexcept = 0;

    virtual void InitScene(void *scene) noexcept {}
    virtual void RegisterGui() noexcept {}
};
}// namespace Pupil