#pragma once

#include <chrono>
namespace Pupil {
class Timer {
public:
    void Start() noexcept {
        m_start_time = std::chrono::system_clock::now();
        m_is_running = true;
    }

    void Stop() noexcept {
        m_end_time = std::chrono::system_clock::now();
        m_is_running = false;
    }

    double ElapsedMilliseconds() noexcept {
        std::chrono::time_point<std::chrono::system_clock> end_time =
            m_is_running ? std::chrono::system_clock::now() : m_end_time;
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_start_time).count();
    }

    double ElapsedSeconds() noexcept {
        return ElapsedMilliseconds() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;
    bool m_is_running = false;
};
}// namespace Pupil