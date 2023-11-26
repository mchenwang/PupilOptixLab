#pragma once

#include "stream.h"
#include "util/timer.h"

namespace Pupil::cuda {
    class GpuTimer : public Pupil::Timer {
    public:
        GpuTimer(const util::CountableRef<Stream>& stream) noexcept;
        ~GpuTimer() noexcept;

        virtual void Start() noexcept override;
        virtual void Stop() noexcept override;

        virtual bool  TryGetElapsedMilliseconds(float& milliseconds) noexcept override;
        virtual float GetElapsedMilliseconds() noexcept override;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };

    class TracedGpuTimer : public Pupil::Timer {
    public:
        TracedGpuTimer(const util::CountableRef<Stream>& stream) noexcept;
        ~TracedGpuTimer() noexcept;

        virtual void Start() noexcept override;
        virtual void Stop() noexcept override;

        virtual bool  TryGetElapsedMilliseconds(float& milliseconds) noexcept override;
        virtual float GetElapsedMilliseconds() noexcept override;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil::cuda