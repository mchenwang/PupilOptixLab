#include "timer.h"

#include <atomic>

namespace Pupil {
    struct TracedCpuTimer::Impl {
        std::chrono::time_point<std::chrono::system_clock> start_time;

        std::atomic_uint64_t elapsed_microseconds;

        std::atomic_bool start_flag;
        std::atomic_bool stop_flag;
    };

    TracedCpuTimer::TracedCpuTimer() noexcept {
        m_impl = new Impl();
        // m_impl->start_flag = false;
        m_impl->stop_flag = false;
    }

    TracedCpuTimer::~TracedCpuTimer() noexcept {
        delete m_impl;
    }

    void TracedCpuTimer::Start() noexcept {
        m_impl->start_time = std::chrono::system_clock::now();
        // m_impl->start_flag = true;
    }

    void TracedCpuTimer::Stop() noexcept {
        m_impl->elapsed_microseconds.store(std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::system_clock::now() - m_impl->start_time)
                                               .count());
        m_impl->stop_flag = true;
    }

    bool TracedCpuTimer::TryGetElapsedMilliseconds(float& milliseconds) noexcept {
        if (m_impl->stop_flag) {
            milliseconds = m_impl->elapsed_microseconds.load() / 1000.f;
            // m_impl->start_flag = false;
            m_impl->stop_flag = false;
            return true;
        }
        return false;
    }

    float TracedCpuTimer::GetElapsedMilliseconds() noexcept {
        return m_impl->elapsed_microseconds.load() / 1000.f;
    }
}// namespace Pupil