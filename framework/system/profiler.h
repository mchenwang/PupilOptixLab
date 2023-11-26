#pragma once

#include "util/util.h"
#include "util/timer.h"

#include "cuda/stream.h"

namespace Pupil {
    class Profiler : public util::Singleton<Profiler> {
    public:
        enum class EType {
            Cpu,
            GpuCuda
        };

        struct Entry {
            const float* datas;
            int          count;
            int          offset;
        };

        Profiler() noexcept;
        ~Profiler() noexcept;

        void Run() noexcept;

        Timer* AllocTimer(std::string_view name, int buffer_size = 120) noexcept;
        Timer* AllocTimer(std::string_view name, const util::CountableRef<cuda::Stream>& stream, int buffer_size = 120) noexcept;

        void Enable(std::string_view name) noexcept;
        void Disable(std::string_view name) noexcept;

        Entry GetEntry(std::string_view name) noexcept;

        void ShowPlot(std::string_view name) noexcept;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace Pupil