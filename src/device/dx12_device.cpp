#include "dx12_device.h"
#include "d3dx12.h"

#include <d3dcompiler.h>

#ifdef _DEBUG
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif

using namespace device;
using Microsoft::WRL::ComPtr;

namespace {

void EnableDebugLayer() noexcept;
bool CheckTearingSupport() noexcept;

}// namespace

DX12::DX12(uint32_t w, uint32_t h, HWND hWnd) noexcept {
    m_frame_w = w;
    m_frame_h = h;

    EnableDebugLayer();
    CreateDevice();
    CreateDescHeap();
    CreateCmdContext();
    CreateSwapchain(hWnd);
}

DX12::~DX12() noexcept {
    Flush();
}

void DX12::CreateDevice() noexcept {
    if (m_adapter == nullptr) CreateAdapter();
    assert(m_adapter);

    StopIfFailed(D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)));
    // Enable debug messages in debug mode.
#ifdef _DEBUG
    ComPtr<ID3D12InfoQueue> info_queue;
    if (SUCCEEDED(device.As(&info_queue))) {
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
        // Suppress whole categories of messages
        //D3D12_MESSAGE_CATEGORY Categories[] = {};

        // Suppress messages based on their severity level
        D3D12_MESSAGE_SEVERITY severities[] = { D3D12_MESSAGE_SEVERITY_INFO };

        // Suppress individual messages by their ID
        D3D12_MESSAGE_ID deny_ids[] = {
            D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,// I'm really not sure how to avoid this message.
            D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                      // This warning occurs when using capture frame while graphics debugging.
            D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                    // This warning occurs when using capture frame while graphics debugging.

            // Workarounds for debug layer issues on hybrid-graphics systems
            D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_WRONGSWAPCHAINBUFFERREFERENCE,
            D3D12_MESSAGE_ID_RESOURCE_BARRIER_MISMATCHING_COMMAND_LIST_TYPE,
        };

        D3D12_INFO_QUEUE_FILTER filter = {};
        //filter.DenyList.NumCategories = _countof(Categories);
        //filter.DenyList.pCategoryList = Categories;
        filter.DenyList.NumSeverities = _countof(severities);
        filter.DenyList.pSeverityList = severities;
        filter.DenyList.NumIDs = _countof(deny_ids);
        filter.DenyList.pIDList = deny_ids;

        StopIfFailed(info_queue->PushStorageFilter(&filter));
    }
#endif
}

void DX12::CreateAdapter() noexcept {
    ComPtr<IDXGIFactory4> dxgi_factory;
    UINT create_factory_flags = 0;
#if defined(_DEBUG)
    create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
    StopIfFailed(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgi_factory)));

    ComPtr<IDXGIAdapter1> dxgi_adapter1;
    ComPtr<IDXGIAdapter4> dxgi_adapter4;

    if (m_use_warp) {
        StopIfFailed(dxgi_factory->EnumWarpAdapter(IID_PPV_ARGS(&dxgi_adapter1)));
        StopIfFailed(dxgi_adapter1.As(&dxgi_adapter4));
    } else {
        ComPtr<IDXGIFactory6> factory6;
        if (SUCCEEDED(dxgi_factory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
            for (
                UINT adapter_index = 0;
                SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                    adapter_index,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,// Find adapter with the highest performance
                    IID_PPV_ARGS(&dxgi_adapter1)));
                ++adapter_index) {
                DXGI_ADAPTER_DESC1 dxgi_adapter_desc1;
                dxgi_adapter1->GetDesc1(&dxgi_adapter_desc1);
                if (dxgi_adapter_desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                    // Don't select the Basic Render Driver adapter.
                    // If you want a software adapter, pass in "/warp" on the command line.
                    continue;
                }

                // Check to see whether the adapter supports Direct3D 12, but don't create the
                // actual device yet.
                if (SUCCEEDED(D3D12CreateDevice(dxgi_adapter1.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                    StopIfFailed(dxgi_adapter1.As(&dxgi_adapter4));
                    break;
                }
            }
        }
    }

    m_adapter = dxgi_adapter4;
}

void DX12::CreateDescHeap() noexcept {
    // rtv
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.NumDescriptors = DX12::NUM_OF_FRAMES;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask = 0;
        StopIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtv_heap)));

        auto size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        auto rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t i = 0; i < DX12::NUM_OF_FRAMES; i++) {
            m_back_buffer_handles[i] = rtv_handle;
            rtv_handle.ptr += size;
        }
    }
    // srv
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // one is used for imgui, and the other is used for rendering results
        desc.NumDescriptors = 1 + DX12::NUM_OF_FRAMES;
        StopIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&srv_heap)));
    }
}

void DX12::CreateSwapchain(HWND hWnd) noexcept {
    m_tearing_supported = CheckTearingSupport();
    ComPtr<IDXGIFactory4> dxgi_factory;
    UINT create_factory_flags = 0;
#if defined(_DEBUG)
    create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif

    StopIfFailed(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgi_factory)));
    DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {};
    swapchain_desc.Width = m_frame_w;
    swapchain_desc.Height = m_frame_h;
    swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapchain_desc.Stereo = FALSE;
    swapchain_desc.SampleDesc = { 1, 0 };
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.BufferCount = DX12::NUM_OF_FRAMES;
    swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    // It is recommended to always allow tearing if tearing support is available.
    swapchain_desc.Flags = m_tearing_supported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    ComPtr<IDXGISwapChain1> swapchain;
    StopIfFailed(dxgi_factory->CreateSwapChainForHwnd(
        cmd_queue.Get(),
        hWnd,
        &swapchain_desc,
        nullptr,
        nullptr,
        &swapchain));

    StopIfFailed(dxgi_factory->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER));
    StopIfFailed(swapchain.As(&m_swapchain));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget();
}

ComPtr<ID3D12CommandAllocator> DX12::GetCmdAllocator() noexcept {
    ComPtr<ID3D12CommandAllocator> allocator = nullptr;
    if (!m_context_pool.empty()) {
        CmdContext &front = m_context_pool.front();
        if (m_fence->GetCompletedValue() >= front.fence_value) {
            allocator = front.allocator;
            m_context_pool.pop();
            StopIfFailed(allocator->Reset());
        }
    }

    if (allocator == nullptr)
        StopIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));

    return allocator;
}

ComPtr<ID3D12GraphicsCommandList> DX12::GetCmdList() noexcept {
    ComPtr<ID3D12CommandAllocator> allocator = GetCmdAllocator();
    ComPtr<ID3D12GraphicsCommandList> cmd_list = nullptr;
    if (!m_command_lists.empty()) {
        cmd_list = m_command_lists.front();
        m_command_lists.pop();
        StopIfFailed(cmd_list->Reset(allocator.Get(), nullptr));
    } else {
        StopIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&cmd_list)));
    }

    StopIfFailed(cmd_list->SetPrivateDataInterface(__uuidof(ID3D12CommandAllocator), allocator.Get()));

    return cmd_list;
}

void DX12::CreateCmdContext() noexcept {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    StopIfFailed(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&cmd_queue)));

    global_fence_value = 0;
    StopIfFailed(device->CreateFence(
        global_fence_value,
        D3D12_FENCE_FLAG_SHARED,// semaphore for cuda
        IID_PPV_ARGS(&m_fence)));
    m_fence_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
}

void DX12::Flush() noexcept {
    uint64_t fence_value = ++global_fence_value;
    cmd_queue->Signal(m_fence.Get(), fence_value);

    if (m_fence->GetCompletedValue() < fence_value) {
        m_fence->SetEventOnCompletion(fence_value, m_fence_event);
        ::WaitForSingleObject(m_fence_event, DWORD_MAX);
    }
}

uint64_t DX12::ExecuteCommandLists(ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    cmd_list->Close();

    ID3D12CommandAllocator *allocator;
    UINT dataSize = sizeof(allocator);
    StopIfFailed(cmd_list->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &allocator));

    ID3D12CommandList *const p_cmd_lists[] = { cmd_list.Get() };
    cmd_queue->ExecuteCommandLists(1, p_cmd_lists);

    uint64_t fence_value = ++global_fence_value;
    cmd_queue->Signal(m_fence.Get(), fence_value);
    m_context_pool.emplace(allocator, fence_value);
    m_command_lists.emplace(cmd_list);
    allocator->Release();

    return fence_value;
}

uint64_t DX12::Present(Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    auto &back_buffer = m_back_buffers[m_current_index];
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    // execute command
    auto fence_value = ExecuteCommandLists(cmd_list);

    uint32_t sync_interval = m_v_sync ? 1 : 0;
    uint32_t present_flags = (m_tearing_supported && !m_v_sync) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    StopIfFailed(m_swapchain->Present(sync_interval, present_flags));

    return fence_value;
}

void DX12::MoveToNextFrame() noexcept {
    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    if (m_fence->GetCompletedValue() < m_frame_fence_values[m_current_index]) {
        m_fence->SetEventOnCompletion(m_frame_fence_values[m_current_index], m_fence_event);
        ::WaitForSingleObject(m_fence_event, DWORD_MAX);
    }
}

void DX12::UpdateRenderTarget() noexcept {
    for (uint32_t i = 0; i < DX12::NUM_OF_FRAMES; i++) {
        ComPtr<ID3D12Resource> backbuffer = nullptr;
        m_swapchain->GetBuffer(i, IID_PPV_ARGS(&backbuffer));
        backbuffer->SetName((L"back buffer " + std::to_wstring(i)).data());
        device->CreateRenderTargetView(backbuffer.Get(), nullptr, m_back_buffer_handles[i]);
        m_back_buffers[i] = backbuffer;
    }
}

void DX12::Resize(uint32_t w, uint32_t h) noexcept {
    Flush();
    m_frame_w = w;
    m_frame_h = h;

    for (uint32_t i = 0; i < NUM_OF_FRAMES; i++) {
        m_back_buffers[i].Reset();
    }

    DXGI_SWAP_CHAIN_DESC desc{};
    StopIfFailed(m_swapchain->GetDesc(&desc));
    StopIfFailed(m_swapchain->ResizeBuffers(NUM_OF_FRAMES, w, h, desc.BufferDesc.Format, desc.Flags));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget();
}

namespace {
void EnableDebugLayer() noexcept {
#ifdef _DEBUG
    // Always enable the debug layer before doing anything DX12 related
    // so all possible errors generated while creating DX12 objects
    // are caught by the debug layer.
    ComPtr<ID3D12Debug> debug_interface;
    StopIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_interface)));
    debug_interface->EnableDebugLayer();
    ComPtr<IDXGIInfoQueue> dxgi_info_queue;
    if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(dxgi_info_queue.GetAddressOf())))) {
        //m_dxgiFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;

        dxgi_info_queue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
        dxgi_info_queue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);

        DXGI_INFO_QUEUE_MESSAGE_ID hide[] = {
            80 /* IDXGISwapChain::GetContainingOutput: The swapchain's adapter does not control the output on which the swapchain's window resides. */,
        };
        DXGI_INFO_QUEUE_FILTER filter = {};
        filter.DenyList.NumIDs = static_cast<UINT>(std::size(hide));
        filter.DenyList.pIDList = hide;
        dxgi_info_queue->AddStorageFilterEntries(DXGI_DEBUG_DXGI, &filter);
    }
#endif
}

bool CheckTearingSupport() noexcept {
    BOOL allow_tearing = FALSE;

    // Rather than create the DXGI 1.5 factory interface directly, we create the
    // DXGI 1.4 interface and query for the 1.5 interface. This is to enable the
    // graphics debugging tools which will not support the 1.5 factory interface
    // until a future update.
    ComPtr<IDXGIFactory4> factory4;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4)))) {
        ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory4.As(&factory5))) {
            if (FAILED(factory5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &allow_tearing, sizeof(allow_tearing)))) {
                allow_tearing = FALSE;
            }
        }
    }

    return allow_tearing == TRUE;
}

}// namespace