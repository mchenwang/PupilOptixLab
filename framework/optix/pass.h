#pragma once

#include "pipeline.h"
#include "sbt.h"

#include "cuda/stream.h"

#include <memory>

namespace Pupil::optix {
    class Pass {
    public:
        Pass(size_t launch_parameter_size, util::CountableRef<cuda::Stream> stream = nullptr) noexcept;
        virtual ~Pass() noexcept;

        void Run(CUdeviceptr param_cuda_memory, unsigned int launch_w, unsigned int launch_h, unsigned int launch_d = 1) noexcept;

        void Synchronize() noexcept;

        auto GetStream() const noexcept { return m_stream; }

    protected:
        util::CountableRef<cuda::Stream> m_stream;
        std::unique_ptr<Pipeline>        m_pipeline;
        std::unique_ptr<SBT>             m_sbt;
        size_t                           m_launch_parameter_size = 0;
    };
}// namespace Pupil::optix