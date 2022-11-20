#pragma once

#include "common/util.h"

#include <d3d12.h>
#include <wrl.h>
#include <dxgi1_6.h>

namespace gui {
class Backend : public util::Singleton<Backend> {
private:
public:
    constexpr static uint32_t NUM_OF_FRAMES = 3;
    Microsoft::WRL::ComPtr<ID3D12Device> device;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srv_heap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtv_heap;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;

    void Flush() noexcept;

    void Init() noexcept;
    void Destroy() noexcept;

    void Resize(uint32_t w, uint32_t h) noexcept;

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetCmdList() noexcept;

    void RenderScreen(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;
    void Present(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;
};
}// namespace gui