#include "timer.h"
#include "check.h"

#include <atomic>

namespace Pupil::cuda {
    struct GpuTimer::Impl {
        util::CountableRef<Stream> stream;
        std::unique_ptr<Event>     start;
        std::unique_ptr<Event>     stop;
    };

    GpuTimer::GpuTimer(const util::CountableRef<Stream>& stream) noexcept
        : Timer() {
        m_impl         = new Impl();
        m_impl->stream = stream;
        m_impl->start  = std::make_unique<cuda::Event>(true);
        m_impl->stop   = std::make_unique<cuda::Event>(true);
    }

    GpuTimer::~GpuTimer() noexcept {
        delete m_impl;
    }

    void GpuTimer::Start() noexcept {
        m_impl->start->Record(m_impl->stream.Get());
    }

    void GpuTimer::Stop() noexcept {
        m_impl->stop->Record(m_impl->stream.Get());
    }

    bool GpuTimer::TryGetElapsedMilliseconds(float& milliseconds) noexcept {
        if (m_impl->start->IsRecorded()) {
            if (m_impl->stop->IsRecorded() && m_impl->stop->IsCompleted()) {
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, *m_impl->start, *m_impl->stop));
                return true;
            }
        }
        return false;
    }

    float GpuTimer::GetElapsedMilliseconds() noexcept {
        float milliseconds = 0.f;
        if (m_impl->stop->IsRecorded()) {
            m_impl->stop->Synchronize();
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, *m_impl->start, *m_impl->stop));
        }
        return milliseconds;
    }

    struct TracedGpuTimer::Impl {
        util::CountableRef<Stream> stream;

        struct
        {
            std::atomic_bool       stop_flag;
            std::unique_ptr<Event> start;
            std::unique_ptr<Event> stop;
        } flip[2];
        std::atomic_int  flip_index;
        std::atomic_bool flip_flag;
        std::atomic_bool first_flag;
    };

    TracedGpuTimer::TracedGpuTimer(const util::CountableRef<Stream>& stream) noexcept
        : Timer() {
        m_impl                    = new Impl();
        m_impl->stream            = stream;
        m_impl->flip[0].start     = std::make_unique<cuda::Event>(true);
        m_impl->flip[0].stop      = std::make_unique<cuda::Event>(true);
        m_impl->flip[1].start     = std::make_unique<cuda::Event>(true);
        m_impl->flip[1].stop      = std::make_unique<cuda::Event>(true);
        m_impl->flip[0].stop_flag = false;
        m_impl->flip[1].stop_flag = false;
        m_impl->flip_index        = 0;
        m_impl->first_flag        = true;
    }

    TracedGpuTimer::~TracedGpuTimer() noexcept {
        delete m_impl;
    }

    void TracedGpuTimer::Start() noexcept {
        if (m_impl->flip_flag) {
            m_impl->flip_index.fetch_xor(1);
            m_impl->flip_flag = false;
        }

        m_impl->flip[m_impl->flip_index].start->Record(m_impl->stream.Get());
    }

    void TracedGpuTimer::Stop() noexcept {
        m_impl->flip[m_impl->flip_index].stop_flag = true;
        m_impl->flip[m_impl->flip_index].stop->Record(m_impl->stream.Get());
    }

    bool TracedGpuTimer::TryGetElapsedMilliseconds(float& milliseconds) noexcept {
        if (m_impl->first_flag) {
            m_impl->first_flag = false;
            m_impl->flip_flag  = true;
            return false;
        }

        int index = m_impl->flip_index.load() ^ 1;
        if (m_impl->flip[index].stop_flag) {
            if (m_impl->flip[index].stop->IsCompleted()) {
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, *m_impl->flip[index].start, *m_impl->flip[index].stop));
                m_impl->flip[index].stop_flag = false;
                m_impl->flip_flag             = true;
                return true;
            }
        }
        return false;
    }

    float TracedGpuTimer::GetElapsedMilliseconds() noexcept {
        float milliseconds = 0.f;
        if (m_impl->flip[m_impl->flip_index].stop_flag) {
            m_impl->flip[m_impl->flip_index].stop->Synchronize();
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, *m_impl->flip[m_impl->flip_index].start, *m_impl->flip[m_impl->flip_index].stop));
        }
        return milliseconds;
    }
}// namespace Pupil::cuda