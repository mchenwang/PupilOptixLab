#pragma once

#include "common/util.h"
#include "device/dx12_device.h"

#include <memory>

namespace device {
struct CudaDx12SharedTexture;
struct SharedFrameResource;
}

namespace gui {
class Backend : public util::Singleton<Backend> {
private:
    std::unique_ptr<device::DX12> m_backend;

public:
    struct {
        device::CudaDx12SharedTexture *src;
        Microsoft::WRL::ComPtr<ID3D12Resource> screen_texture = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE screen_cpu_srv_handle;
        D3D12_GPU_DESCRIPTOR_HANDLE screen_gpu_srv_handle;
    } frames[device::DX12::NUM_OF_FRAMES];
    
    void SetScreenResource(device::SharedFrameResource *) noexcept;

    auto GetDevice() noexcept { return m_backend.get(); }
    void Flush() noexcept { m_backend->Flush(); }

    void Init() noexcept;
    void Destroy() noexcept;

    void Resize(uint32_t w, uint32_t h) noexcept { m_backend->Resize(w, h); }

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetCmdList() noexcept { return m_backend->GetCmdList(); }

    void RenderScreen(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;
    void Present(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;
};
}// namespace gui