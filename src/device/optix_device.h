#pragma once

#include "dx12_device.h"

#include "optix_wrap/sbt.h"
#include "optix_wrap/pipeline.h"

#include <memory>
#include <any>

namespace optix_wrap {
struct PipelineDesc;
struct Pipeline;
struct RenderObject;
}// namespace optix_wrap

namespace scene {
class Scene;
}

namespace device {
struct CudaDx12SharedTexture {
    Microsoft::WRL::ComPtr<ID3D12Resource> dx12_resource;
    cudaExternalMemory_t cuda_ext_memory;
    void *cuda_buffer_ptr;
    //cudaSurfaceObject_t cuda_surf_obj;
    uint64_t fence_value;
};

struct SharedFrameResource {
    std::unique_ptr<CudaDx12SharedTexture> frame[DX12::NUM_OF_FRAMES];
};

class Optix {
public:
    uint32_t cuda_device_id = 0;
    uint32_t cuda_node_mask = 0;

    cudaStream_t cuda_stream = nullptr;
    cudaExternalSemaphore_t cuda_semaphore = nullptr;

    OptixDeviceContext context = nullptr;

    OptixTraversableHandle ias_handle = 0;
    CUdeviceptr ias_buffer = 0;

    Optix() = delete;
    Optix(DX12 *dx12_backend) noexcept;
    ~Optix() noexcept;

    [[nodiscard]] SharedFrameResource *GetSharedFrameResource() noexcept;
    void ClearSharedFrameResource() noexcept { m_frame_resource.reset(); }

    void InitPipeline(const optix_wrap::PipelineDesc &) noexcept;

    template<optix_wrap::SBTTypes T>
    void InitSBT(const optix_wrap::SBTDesc<T> &) noexcept;
    void InitScene(scene::Scene *scene) noexcept;

    void Run(void *params, size_t params_size, void **frame_buffer) noexcept;

private:
    DX12 *m_dx12_backend = nullptr;
    std::unique_ptr<SharedFrameResource> m_frame_resource = nullptr;
    std::unique_ptr<optix_wrap::Pipeline> pipeline = nullptr;
    std::any sbt{};
    OptixShaderBindingTable *m_sbt_ptr = nullptr;

    std::vector<std::unique_ptr<optix_wrap::RenderObject>> m_ros;

    size_t m_cuda_params_size = 0;
    void *m_cuda_params = nullptr;

    void InitCuda() noexcept;
    void CreateTopLevelAccel(std::vector<std::unique_ptr<optix_wrap::RenderObject>> &) noexcept;

    [[nodiscard]] std::unique_ptr<CudaDx12SharedTexture>
    CreateSharedResourceWithDX12() noexcept;
};
}// namespace device

namespace device {

template<optix_wrap::SBTTypes T>
void Optix::InitSBT(const optix_wrap::SBTDesc<T> &desc) noexcept {
    //sbt = std::make_unique<decltype(sbt)>();
    sbt = optix_wrap::SBT<T>{};
    auto o = std::any_cast<optix_wrap::SBT<T>>(&sbt);
    {
        optix_wrap::BindingInfo<typename T::RayGenDataType> rg_data;
        typename decltype(rg_data)::Pair data{
            .program = pipeline->FindProgram(desc.ray_gen_data.program_name),
            .data = desc.ray_gen_data.data
        };
        rg_data.datas.push_back(data);
        o->SetRayGenData(rg_data);
    }
    {
        optix_wrap::BindingInfo<typename T::HitGroupDataType> hit_datas{};
        for (auto &hit_data : desc.hit_datas) {
            typename decltype(hit_datas)::Pair data{
                .program = pipeline->FindProgram(hit_data.program_name),
                .data = hit_data.data
            };
            hit_datas.datas.push_back(data);
        }
        o->SetHitGroupData(hit_datas);
    }
    {
        optix_wrap::BindingInfo<typename T::MissDataType> miss_datas{};
        for (auto &miss_data : desc.miss_datas) {
            typename decltype(miss_datas)::Pair data{
                .program = pipeline->FindProgram(miss_data.program_name),
                .data = miss_data.data
            };
            miss_datas.datas.push_back(data);
        }
        o->SetMissData(miss_datas);
    }

    m_sbt_ptr = &o->sbt;
}
}// namespace device