#pragma once

#include "sbt.h"
#include "pipeline.h"

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

public:
    Pass(const OptixDeviceContext device_context, const cudaStream_t cuda_stream) noexcept
        : m_device_context(device_context), m_cuda_stream(cuda_stream) {}

    ~Pass() noexcept {
        CUDA_FREE(m_param_cuda_memory);
        m_sbt.reset();
        m_pipeline.reset();
    }

    void InitPipeline(const PipelineDesc &desc) noexcept {
        m_pipeline = std::make_unique<Pipeline>(m_device_context, desc);
    }

    void InitSBT(const SBTDesc<T> &desc) noexcept {
        m_sbt = std::make_unique<SBT<T>>();
        {
            BindingInfo<typename T::RayGenDataType> rg_data;
            typename decltype(rg_data)::Pair data{
                .program = m_pipeline->FindProgram(desc.ray_gen_data.program_name),
                .data = desc.ray_gen_data.data
            };
            rg_data.datas.push_back(data);
            m_sbt->SetRayGenData(rg_data);
        }
        {
            BindingInfo<typename T::HitGroupDataType> hit_datas{};
            for (auto &hit_data : desc.hit_datas) {
                typename decltype(hit_datas)::Pair data{
                    .program = m_pipeline->FindProgram(hit_data.program_name),
                    .data = hit_data.data
                };
                hit_datas.datas.push_back(data);
            }
            m_sbt->SetHitGroupData(hit_datas);
        }
        {
            BindingInfo<typename T::MissDataType> miss_datas{};
            for (auto &miss_data : desc.miss_datas) {
                typename decltype(miss_datas)::Pair data{
                    .program = m_pipeline->FindProgram(miss_data.program_name),
                    .data = miss_data.data
                };
                miss_datas.datas.push_back(data);
            }
            m_sbt->SetMissData(miss_datas);
        }
    }

    void Run(const LaunchParamT &params, const unsigned int launch_w, const unsigned int launch_h) noexcept {
        if (m_param_cuda_memory == nullptr)
            CUDA_CHECK(cudaMalloc(&m_param_cuda_memory, sizeof(LaunchParamT)));

        CUDA_CHECK(cudaMemcpyAsync(
            m_param_cuda_memory,
            &params, sizeof(LaunchParamT),
            cudaMemcpyHostToDevice, m_cuda_stream));

        OPTIX_CHECK(optixLaunch(
            *m_pipeline.get(),
            m_cuda_stream,
            reinterpret_cast<CUdeviceptr>(m_param_cuda_memory),
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