#pragma once

#include "pipeline.h"
#include "sbt.h"

#include <memory>

namespace Pupil::optix {
template<SBTTypes T, typename LaunchParamT>
class Pass {
private:
    std::unique_ptr<SBT<T>> m_sbt = nullptr;
    std::unique_ptr<Pipeline> m_pipeline = nullptr;

    void *m_param_cuda_memory = nullptr;

    const OptixDeviceContext m_device_context;
    const cudaStream_t m_cuda_stream;
    bool m_use_inner_param = true;

public:
    Pass(const OptixDeviceContext device_context, const cudaStream_t cuda_stream, const bool use_inner_param = true) noexcept
        : m_device_context(device_context), m_cuda_stream(cuda_stream), m_use_inner_param(use_inner_param) {}

    ~Pass() noexcept {
        if (m_use_inner_param)
            CUDA_FREE(m_param_cuda_memory);
        m_sbt.reset();
        m_pipeline.reset();
    }

    void InitPipeline(const PipelineDesc &desc) noexcept {
        m_pipeline = std::make_unique<Pipeline>(desc);
    }

    void InitSBT(const SBTDesc<T> &desc) noexcept {
        m_sbt = std::make_unique<SBT<T>>(desc, m_pipeline.get());
    }

    void SetExternalLaunchParamPtr(CUdeviceptr cuda_memory) noexcept {
        m_param_cuda_memory = reinterpret_cast<void *>(cuda_memory);
        m_use_inner_param = false;
    }

    CUdeviceptr GetLaunchParamPtr() noexcept {
        if (m_use_inner_param) {
            if (m_param_cuda_memory == nullptr)
                CUDA_CHECK(cudaMalloc(&m_param_cuda_memory, sizeof(LaunchParamT)));
        } else {
            assert(m_param_cuda_memory != nullptr && "CUDA memory for OptiX launch parameters should be allocated");
        }

        return reinterpret_cast<CUdeviceptr>(m_param_cuda_memory);
    }

    void Run(CUdeviceptr param_cuda_memory, const unsigned int launch_w, const unsigned int launch_h) noexcept {
        OPTIX_CHECK(optixLaunch(
            *m_pipeline.get(),
            m_cuda_stream,
            param_cuda_memory,
            sizeof(LaunchParamT),
            &m_sbt->sbt,
            launch_w,
            launch_h,
            1// launch depth
            ));
    }

    void Run(const LaunchParamT &params, const unsigned int launch_w, const unsigned int launch_h) noexcept {
        auto param_cuda_memory = GetLaunchParamPtr();

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(param_cuda_memory),
            &params, sizeof(LaunchParamT),
            cudaMemcpyHostToDevice, m_cuda_stream));

        OPTIX_CHECK(optixLaunch(
            *m_pipeline.get(),
            m_cuda_stream,
            param_cuda_memory,
            sizeof(LaunchParamT),
            &m_sbt->sbt,
            launch_w,
            launch_h,
            1// launch depth
            ));
    }

    void Synchronize() noexcept {
        CUDA_CHECK(cudaStreamSynchronize(m_cuda_stream));
    }
};
}// namespace Pupil::optix