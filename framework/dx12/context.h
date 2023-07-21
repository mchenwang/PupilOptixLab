#pragma once

#include "util/util.h"

#include <queue>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <winrt/base.h>
#include <comdef.h>
#include <mutex>

namespace Pupil::DirectX {

inline void StopIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        _com_error err(hr);
        OutputDebugString(err.ErrorMessage());
        assert(false);
    }
}

class Context : public Pupil::util::Singleton<Context> {
public:
    constexpr static uint32_t FRAMES_NUM = 3;

    winrt::com_ptr<ID3D12Device> device;
    winrt::com_ptr<IDXGIAdapter4> adapter;

    // 0    ----    imgui font
    // 1    ----    output flip texture 0 for imgui
    // 2    ----    output flip buffer 0
    // 3    ----    output flip texture 1 for imgui
    // 4    ----    output flip buffer 1
    // 5... ----    for custom
    winrt::com_ptr<ID3D12DescriptorHeap> srv_heap;
    constexpr static uint32_t SRV_NUM = 20;
    // 0    ----    swapchain back buffer 0
    // 1    ----    swapchain back buffer 1
    // 2    ----    swapchain back buffer 2
    // 3    ----    output flip texture 0
    // 4    ----    output flip texture 1
    // 5... ----    for custom
    winrt::com_ptr<ID3D12DescriptorHeap> rtv_heap;
    constexpr static uint32_t RTV_NUM = 20;

    winrt::com_ptr<ID3D12CommandQueue> cmd_queue;
    uint64_t global_fence_value = 0;

    void Init(uint32_t w, uint32_t h, HWND wnd_handle) noexcept;
    void Destroy() noexcept;
    static void ReportLiveObjects();

    void Flush() noexcept;
    void Resize(uint32_t w, uint32_t h) noexcept;

    struct Frame {
        winrt::com_ptr<ID3D12Resource> buffer;
        D3D12_CPU_DESCRIPTOR_HANDLE handle;
    };
    [[nodiscard]] Frame GetCurrentFrame() noexcept {
        std::scoped_lock lock{ m_flip_model_mutex };
        return Frame{ m_back_buffers[m_current_index], m_back_buffer_handles[m_current_index] };
    }
    [[nodiscard]] uint32_t GetCurrentFrameIndex() noexcept { return m_current_index; }
    [[nodiscard]] uint64_t GetGlobalFenceValue() noexcept { return global_fence_value; }

    [[nodiscard]] winrt::com_ptr<ID3D12GraphicsCommandList> GetCmdList() noexcept;
    uint64_t ExecuteCommandLists(winrt::com_ptr<ID3D12GraphicsCommandList>) noexcept;

    void StartRenderScreen(winrt::com_ptr<ID3D12GraphicsCommandList>) noexcept;
    void Present(winrt::com_ptr<ID3D12GraphicsCommandList>) noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }

private:
    uint32_t m_frame_w = 1;
    uint32_t m_frame_h = 1;

    bool m_init_flag = false;

    bool m_use_warp = false;
    bool m_v_sync = true;
    bool m_tearing_supported = false;
    winrt::com_ptr<IDXGISwapChain4> m_swapchain;

    HANDLE m_fence_event{};
    winrt::com_ptr<ID3D12Fence> m_fence;
    std::array<uint64_t, Context::FRAMES_NUM> m_frame_fence_values{};

    struct CmdContext {
        winrt::com_ptr<ID3D12CommandAllocator> allocator;
        uint64_t fence_value;
    };

    std::queue<winrt::com_ptr<ID3D12GraphicsCommandList>> m_command_lists;
    std::queue<CmdContext> m_context_pool;

    uint32_t m_current_index;
    std::mutex m_flip_model_mutex;
    std::array<winrt::com_ptr<ID3D12Resource>, Context::FRAMES_NUM> m_back_buffers{};
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, Context::FRAMES_NUM> m_back_buffer_handles{};

    void CreateAdapter() noexcept;
    void CreateDevice() noexcept;
    void CreateDescHeap() noexcept;
    void CreateCmdContext() noexcept;
    void CreateSwapchain(HWND hWnd) noexcept;
    void UpdateRenderTarget() noexcept;

    [[nodiscard]] winrt::com_ptr<ID3D12CommandAllocator> GetCmdAllocator() noexcept;
};
}// namespace Pupil::DirectX