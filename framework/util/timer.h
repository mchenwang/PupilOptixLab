#pragma once

#include <chrono>
#include <atomic>

namespace Pupil {
    class Timer {
    public:
        virtual void Start() noexcept = 0;
        virtual void Stop() noexcept  = 0;

        virtual bool  TryGetElapsedMilliseconds(float& milliseconds) noexcept = 0;
        virtual float GetElapsedMilliseconds() noexcept                       = 0;
    };

    class CpuTimer : public Timer {
    public:
        CpuTimer()  = default;
        ~CpuTimer() = default;

        virtual void Start() noexcept override {
            m_start_time = std::chrono::system_clock::now();
            m_is_running.store(true);
        }

        virtual void Stop() noexcept override {
            m_end_time = std::chrono::system_clock::now();
            m_is_running.store(false);
        }

        virtual bool TryGetElapsedMilliseconds(float& milliseconds) noexcept override {
            if (m_is_running.load()) return false;
            milliseconds = GetElapsedMilliseconds();
            return true;
        }

        virtual float GetElapsedMilliseconds() noexcept override {
            std::chrono::time_point<std::chrono::system_clock> end_time =
                m_is_running.load() ? std::chrono::system_clock::now() : m_end_time;
            return std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time).count() / 1000.;
        }

    private:
        std::chrono::time_point<std::chrono::system_clock> m_start_time;
        std::chrono::time_point<std::chrono::system_clock> m_end_time;
        std::atomic_bool                                   m_is_running{false};
    };

    class TracedCpuTimer : public Pupil::Timer {
    public:
        TracedCpuTimer() noexcept;
        ~TracedCpuTimer() noexcept;

        virtual void Start() noexcept override;
        virtual void Stop() noexcept override;

        virtual bool  TryGetElapsedMilliseconds(float& milliseconds) noexcept override;
        virtual float GetElapsedMilliseconds() noexcept override;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil