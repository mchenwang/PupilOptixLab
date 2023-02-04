#pragma once

#include <optix.h>
#include <string>

namespace optix_wrap {
struct Module {
    Module(OptixDeviceContext context, OptixPrimitiveType) noexcept;
    Module(OptixDeviceContext context, std::string_view) noexcept;
    ~Module() noexcept;

    OptixModule module = nullptr;
};
}// namespace optix_wrap