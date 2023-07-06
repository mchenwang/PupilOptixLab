#pragma once

#include "util.h"

#if defined(PUPIL_CUDA)
#include <cstdio>
#elif defined(PUPIL_CPP)
#include "spdlog/spdlog.h"
#endif

namespace Pupil {
class Log : public util::Singleton<Log> {
public:
#if defined(PUPIL_CUDA)
    void Init() noexcept {}
    void Destroy() noexcept {}

    template<typename... T>
    static void Info(T &&...) noexcept {}

    template<typename... T>
    static void Error(T &&...) noexcept {}

    template<typename... T>
    static void Warn(T &&...) noexcept {}
#elif defined(PUPIL_CPP)
    void Init() noexcept {
        spdlog::set_pattern("[%^%l%$] %v");
    }
    void Destroy() noexcept {}

    template<typename... Args>
    static void Info(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
        spdlog::info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Error(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
        spdlog::error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void Warn(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
        spdlog::warn(fmt, std::forward<Args>(args)...);
    }
#else
    void Init() noexcept {}
    void Destroy() noexcept {}

    template<typename... T>
    static void Info(T &&...) noexcept {}

    template<typename... T>
    static void Error(T &&...) noexcept {}

    template<typename... T>
    static void Warn(T &&...) noexcept {}
#endif
};
}// namespace Pupil