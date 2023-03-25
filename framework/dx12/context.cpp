#include "context.h"
#include "d3dx12.h"

#include <d3dcompiler.h>

#ifdef _DEBUG
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif

namespace {

void EnableDebugLayer() noexcept;
bool CheckTearingSupport() noexcept;

}// namespace

namespace Pupil::DirectX {
void Context::Init(uint32_t w, uint32_t h, HWND hWnd) noexcept {
    m_frame_w = w;
    m_frame_h = h;

    EnableDebugLayer();
    CreateDevice();
    CreateDescHeap();
    CreateCmdContext();
    CreateSwapchain(hWnd);

    m_init_flag = true;
}

void Context::Destroy() noexcept {
    Flush();
    ::CloseHandle(m_fence_event);
    m_init_flag = false;
}

void Context::CreateDevice() noexcept {
    if (adapter == nullptr) CreateAdapter();
    assert(adapter);

    StopIfFailed(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_1, winrt::guid_of<ID3D12Device>(), device.put_void()));
    // Enable debug messages in debug mode.
#ifdef _DEBUG
    winrt::com_ptr<ID3D12InfoQueue> info_queue;
    if (SUCCEEDED(device.as(winrt::guid_of<ID3D12InfoQueue>(), info_queue.put_void()))) {
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

void Context::CreateAdapter() noexcept {
    winrt::com_ptr<IDXGIFactory4> dxgi_factory;
    UINT create_factory_flags = 0;
#if defined(_DEBUG)
    create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
    StopIfFailed(CreateDXGIFactory2(create_factory_flags, winrt::guid_of<IDXGIFactory4>(), dxgi_factory.put_void()));

    winrt::com_ptr<IDXGIAdapter1> dxgi_adapter1;
    winrt::com_ptr<IDXGIAdapter4> dxgi_adapter4;

    if (m_use_warp) {
        StopIfFailed(dxgi_factory->EnumWarpAdapter(winrt::guid_of<IDXGIAdapter1>(), dxgi_adapter1.put_void()));
        StopIfFailed(dxgi_adapter1.as(winrt::guid_of<IDXGIAdapter4>(), reinterpret_cast<void **>(dxgi_adapter4.put())));
    } else {
        winrt::com_ptr<IDXGIFactory6> factory6;
        if (SUCCEEDED(dxgi_factory->QueryInterface(winrt::guid_of<IDXGIFactory6>(), factory6.put_void()))) {
            for (
                UINT adapter_index = 0;
                SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                    adapter_index,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,// Find adapter with the highest performance
                    winrt::guid_of<IDXGIAdapter1>(),
                    dxgi_adapter1.put_void()));
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
                if (SUCCEEDED(D3D12CreateDevice(dxgi_adapter1.get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                    StopIfFailed(dxgi_adapter1.as(winrt::guid_of<IDXGIAdapter4>(), dxgi_adapter4.put_void()));
                    break;
                }
            }
        }
    }

    adapter = dxgi_adapter4;
}

void Context::CreateDescHeap() noexcept {
    // rtv
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.NumDescriptors = Context::FRAMES_NUM;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask = 0;
        StopIfFailed(device->CreateDescriptorHeap(&desc, winrt::guid_of<ID3D12DescriptorHeap>(), rtv_heap.put_void()));

        auto size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        auto rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t i = 0; i < Context::FRAMES_NUM; i++) {
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
        desc.NumDescriptors = 1 + Context::FRAMES_NUM;
        StopIfFailed(device->CreateDescriptorHeap(&desc, winrt::guid_of<ID3D12DescriptorHeap>(), srv_heap.put_void()));
    }
}

void Context::CreateSwapchain(HWND hWnd) noexcept {
    m_tearing_supported = CheckTearingSupport();
    winrt::com_ptr<IDXGIFactory4> dxgi_factory;
    UINT create_factory_flags = 0;
#if defined(_DEBUG)
    create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif

    StopIfFailed(CreateDXGIFactory2(create_factory_flags, winrt::guid_of<IDXGIFactory4>(), dxgi_factory.put_void()));
    DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {};
    swapchain_desc.Width = m_frame_w;
    swapchain_desc.Height = m_frame_h;
    swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapchain_desc.Stereo = FALSE;
    swapchain_desc.SampleDesc = { 1, 0 };
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.BufferCount = Context::FRAMES_NUM;
    swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    // It is recommended to always allow tearing if tearing support is available.
    swapchain_desc.Flags = m_tearing_supported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    winrt::com_ptr<IDXGISwapChain1> swapchain;
    StopIfFailed(dxgi_factory->CreateSwapChainForHwnd(
        cmd_queue.get(),
        hWnd,
        &swapchain_desc,
        nullptr,
        nullptr,
        swapchain.put()));

    StopIfFailed(dxgi_factory->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER));
    StopIfFailed(swapchain.as(winrt::guid_of<IDXGISwapChain4>(), m_swapchain.put_void()));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget();
}

winrt::com_ptr<ID3D12CommandAllocator> Context::GetCmdAllocator() noexcept {
    winrt::com_ptr<ID3D12CommandAllocator> allocator = nullptr;
    if (!m_context_pool.empty()) {
        CmdContext &front = m_context_pool.front();
        if (m_fence->GetCompletedValue() >= front.fence_value) {
            allocator = front.allocator;
            m_context_pool.pop();
            StopIfFailed(allocator->Reset());
        }
    }

    if (allocator == nullptr)
        StopIfFailed(device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            winrt::guid_of<ID3D12CommandAllocator>(), allocator.put_void()));

    return allocator;
}

winrt::com_ptr<ID3D12GraphicsCommandList> Context::GetCmdList() noexcept {
    winrt::com_ptr<ID3D12CommandAllocator> allocator = GetCmdAllocator();
    winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list = nullptr;
    if (!m_command_lists.empty()) {
        cmd_list = m_command_lists.front();
        m_command_lists.pop();
        StopIfFailed(cmd_list->Reset(allocator.get(), nullptr));
    } else {
        StopIfFailed(device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.get(), nullptr,
            winrt::guid_of<ID3D12GraphicsCommandList>(), cmd_list.put_void()));
    }

    StopIfFailed(cmd_list->SetPrivateDataInterface(__uuidof(ID3D12CommandAllocator), allocator.get()));

    return cmd_list;
}

void Context::CreateCmdContext() noexcept {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    StopIfFailed(device->CreateCommandQueue(&desc, winrt::guid_of<ID3D12CommandQueue>(), cmd_queue.put_void()));

    global_fence_value = 0;
    StopIfFailed(device->CreateFence(
        global_fence_value,
        D3D12_FENCE_FLAG_SHARED,// semaphore for cuda
        winrt::guid_of<ID3D12Fence>(),
        m_fence.put_void()));
    m_fence_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
}

void Context::Flush() noexcept {
    uint64_t fence_value = ++global_fence_value;
    cmd_queue->Signal(m_fence.get(), fence_value);

    if (m_fence->GetCompletedValue() < fence_value) {
        m_fence->SetEventOnCompletion(fence_value, m_fence_event);
        ::WaitForSingleObject(m_fence_event, 0xffffffffUL);
    }
}

uint64_t Context::ExecuteCommandLists(winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    cmd_list->Close();

    /*ID3D12CommandAllocator *allocator;
    UINT dataSize = sizeof(allocator);
    StopIfFailed(cmd_list->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &allocator));*/

    winrt::com_ptr<ID3D12CommandAllocator> allocator;
    UINT data_size = sizeof(ID3D12CommandAllocator *);
    StopIfFailed(cmd_list->GetPrivateData(__uuidof(ID3D12CommandAllocator), &data_size, allocator.put_void()));

    ID3D12CommandList *const p_cmd_lists[] = { cmd_list.get() };
    cmd_queue->ExecuteCommandLists(1, p_cmd_lists);

    uint64_t fence_value = ++global_fence_value;
    cmd_queue->Signal(m_fence.get(), fence_value);
    m_context_pool.emplace(allocator, fence_value);
    m_command_lists.emplace(cmd_list);
    allocator->Release();

    return fence_value;
}

uint64_t Context::Present(winrt::com_ptr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    auto &back_buffer = m_back_buffers[m_current_index];
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.get();
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

void Context::MoveToNextFrame() noexcept {
    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    if (m_fence->GetCompletedValue() < m_frame_fence_values[m_current_index]) {
        m_fence->SetEventOnCompletion(m_frame_fence_values[m_current_index], m_fence_event);
        ::WaitForSingleObject(m_fence_event, 0xffffffffUL);
    }
}

void Context::UpdateRenderTarget() noexcept {
    for (uint32_t i = 0; i < Context::FRAMES_NUM; i++) {
        winrt::com_ptr<ID3D12Resource> backbuffer = nullptr;
        m_swapchain->GetBuffer(i, winrt::guid_of<ID3D12Resource>(), backbuffer.put_void());
        backbuffer->SetName((L"back buffer " + std::to_wstring(i)).data());
        device->CreateRenderTargetView(backbuffer.get(), nullptr, m_back_buffer_handles[i]);
        m_back_buffers[i] = backbuffer;
    }
}

void Context::Resize(uint32_t w, uint32_t h) noexcept {
    Flush();
    m_frame_w = w;
    m_frame_h = h;

    for (uint32_t i = 0; i < FRAMES_NUM; i++) {
        m_back_buffers[i] = nullptr;
    }

    DXGI_SWAP_CHAIN_DESC desc{};
    StopIfFailed(m_swapchain->GetDesc(&desc));
    StopIfFailed(m_swapchain->ResizeBuffers(FRAMES_NUM, w, h, desc.BufferDesc.Format, desc.Flags));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget();
}
}// namespace Pupil::DirectX

namespace {
void EnableDebugLayer() noexcept {
#ifdef _DEBUG
    // Always enable the debug layer before doing anything DX12 related
    // so all possible errors generated while creating DX12 objects
    // are caught by the debug layer.
    winrt::com_ptr<ID3D12Debug> debug_interface;
    Pupil::DirectX::StopIfFailed(D3D12GetDebugInterface(__uuidof(debug_interface), debug_interface.put_void()));
    debug_interface->EnableDebugLayer();
    winrt::com_ptr<IDXGIInfoQueue> dxgi_info_queue;
    if (SUCCEEDED(DXGIGetDebugInterface1(0, __uuidof(dxgi_info_queue), dxgi_info_queue.put_void()))) {
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
    winrt::com_ptr<IDXGIFactory4> factory4;
    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(factory4), factory4.put_void()))) {
        winrt::com_ptr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory4.as(winrt::guid_of<IDXGIFactory5>(), factory5.put_void()))) {
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