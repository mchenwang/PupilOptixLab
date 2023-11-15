#include "pass.h"
#include "context.h"
#include "check.h"

#include <optix_stubs.h>

namespace Pupil::optix {
    Pass::Pass(size_t launch_parameter_size, util::CountableRef<cuda::Stream> stream) noexcept {
        m_stream   = stream ? stream : util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::Render);
        m_pipeline = std::make_unique<Pipeline>();
        m_sbt      = std::make_unique<SBT>(m_pipeline.get());

        m_launch_parameter_size = launch_parameter_size;
    }

    Pass::~Pass() noexcept {
        m_stream.Reset();
        m_pipeline.reset();
        m_sbt.reset();
    }

    void Pass::Run(CUdeviceptr param_cuda_memory, unsigned int launch_w, unsigned int launch_h, unsigned int launch_d) noexcept {
        OptixShaderBindingTable sbt = *m_sbt.get();
        OPTIX_CHECK(optixLaunch(
            *m_pipeline,
            *m_stream,
            param_cuda_memory,
            m_launch_parameter_size,
            &sbt,
            launch_w,
            launch_h,
            launch_d));
    }

    void Pass::Synchronize() noexcept {
        m_stream->Synchronize();
    }
}// namespace Pupil::optix
