#pragma once

#include <optix.h>
#include <string>
namespace device {
class Optix;
}

namespace optix_wrap {
struct Module {
    Module(device::Optix *, OptixPrimitiveType) noexcept;
    Module(device::Optix *, std::string_view) noexcept;
    ~Module() noexcept;

    OptixModule module = nullptr;
};
}// namespace optix_wrap