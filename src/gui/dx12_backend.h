#pragma once

#include "common/util.h"

#include <d3d12.h>
#include <wrl.h>
#include <dxgi1_6.h>

namespace gui {
class Backend : public util::Singleton<Backend> {
private:
    bool m_use_warp = false;
    Microsoft::WRL::ComPtr<IDXGIAdapter4> m_adapter;

    bool m_v_sync;
    bool m_tearing_supported;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_swapchain;

    void CreateAdapter();
    void CreateDevice();
    void CreateDescHeap();
    void CreateCmdContext();
    void CreateSwapchain();

public:
    constexpr static uint32_t NUM_OF_FRAMES = 3;
    Microsoft::WRL::ComPtr<ID3D12Device> device;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srv_heap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtv_heap;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;

    void Init();
};
}// namespace gui