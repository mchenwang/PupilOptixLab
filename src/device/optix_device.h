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
    Microsoft::WRL::ComPtr<ID3D12Resource> dx12_resource = nullptr;
    cudaExternalMemory_t cuda_ext_memory = nullptr;
    void *cuda_buffer_ptr = nullptr;
    //cudaSurfaceObject_t cuda_surf_obj;
    uint64_t fence_value = 0;

    ~CudaDx12SharedTexture() noexcept;
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

    void InitScene(scene::Scene *scene) noexcept;

private:
    DX12 *m_dx12_backend = nullptr;
    std::unique_ptr<SharedFrameResource> m_frame_resource = nullptr;

    std::vector<std::unique_ptr<optix_wrap::RenderObject>> m_ros;

    void InitCuda() noexcept;
    void CreateTopLevelAccel(std::vector<std::unique_ptr<optix_wrap::RenderObject>> &) noexcept;

    [[nodiscard]] std::unique_ptr<CudaDx12SharedTexture>
    CreateSharedResourceWithDX12() noexcept;
};
}// namespace device
