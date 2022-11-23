#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <driver_types.h>

#include <queue>
#include <array>

#include <comdef.h>

inline void StopIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        _com_error err(hr);
        OutputDebugString(err.ErrorMessage());
        assert(false);
    }
}

namespace device {
class DX12 final {
    friend class Optix;

public:
    constexpr static uint32_t NUM_OF_FRAMES = 3;
    Microsoft::WRL::ComPtr<ID3D12Device> device;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srv_heap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtv_heap;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    uint64_t global_fence_value = 0;

    DX12() = delete;
    DX12(uint32_t w, uint32_t h, HWND hWnd)
    noexcept;
    ~DX12() noexcept = default;

    void Flush() noexcept;
    void Resize(uint32_t w, uint32_t h) noexcept;

    struct Frame {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        D3D12_CPU_DESCRIPTOR_HANDLE handle;
    };
    [[nodiscard]] Frame GetCurrentFrame() noexcept {
        return Frame{m_back_buffers[m_current_index], m_back_buffer_handles[m_current_index]};
    }
    [[nodiscard]] uint32_t GetCurrentFrameIndex() noexcept { return m_current_index; }
    [[nodiscard]] uint64_t GetGlobalFenceValue() noexcept { return global_fence_value; }

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> GetCmdList() noexcept;
    uint64_t ExecuteCommandLists(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;

    [[nodiscard]] uint64_t Present(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>) noexcept;
    void SetCurrentFrameFenceValue(uint64_t fence_value) noexcept { m_frame_fence_values[m_current_index] = fence_value; }
    void MoveToNextFrame() noexcept;

private:
    uint32_t m_frame_w = 1;
    uint32_t m_frame_h = 1;

    bool m_use_warp = false;
    Microsoft::WRL::ComPtr<IDXGIAdapter4> m_adapter;

    bool m_v_sync = false;
    bool m_tearing_supported = false;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_swapchain;

    HANDLE m_fence_event{};
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    std::array<uint64_t, DX12::NUM_OF_FRAMES> m_frame_fence_values{};

    struct CmdContext {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        uint64_t fence_value;
    };

    std::queue<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>> m_command_lists;
    std::queue<CmdContext> m_context_pool;

    uint32_t m_current_index;
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, DX12::NUM_OF_FRAMES> m_back_buffers{};
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, DX12::NUM_OF_FRAMES> m_back_buffer_handles{};

    void CreateAdapter() noexcept;
    void CreateDevice() noexcept;
    void CreateDescHeap() noexcept;
    void CreateCmdContext() noexcept;
    void CreateSwapchain(HWND hWnd) noexcept;
    void UpdateRenderTarget() noexcept;

    [[nodiscard]] Microsoft::WRL::ComPtr<ID3D12CommandAllocator> GetCmdAllocator() noexcept;
};
}// namespace device