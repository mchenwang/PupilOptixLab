#pragma once

#include "util.h"
#include "spdlog/spdlog.h"

namespace Pupil {
class Log : public util::Singleton<Log> {
public:
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
};
}// namespace Pupil