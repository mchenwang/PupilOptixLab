#pragma once

#include "util/util.h"

#include <optix.h>
#include <string>
#include <unordered_map>
#include <memory>

namespace Pupil::optix {
struct Module {
    Module(OptixDeviceContext context, OptixPrimitiveType) noexcept;
    Module(OptixDeviceContext context, std::string_view) noexcept;
    ~Module() noexcept;

    OptixModule optix_module = nullptr;
    operator OptixModule() const noexcept { return optix_module; }
};

enum class EModuleBuiltinType {
    CustomPrimitive,
    RoundQuadraticBsplinePrimitive,
    RoundCubicBsplinePrimitive,
    RoundLinearPrimitive,
    RoundCatmullromPrimitive,
    SpherePrimitive,
    TrianglePrimitive,
    Material,
};

class ModuleManager : public Pupil::util::Singleton<ModuleManager> {
private:
    // std::unordered_map<std::string, std::unique_ptr<Module>, util::StringHash, std::equal_to<>> m_modules;
    std::unordered_map<const char *, std::unique_ptr<Module>> m_modules;

public:
    [[nodiscard]] Module *GetModule(EModuleBuiltinType) noexcept;
    [[nodiscard]] Module *GetModule(std::string_view) noexcept;

    void Clear() noexcept;
};
}// namespace Pupil::optix