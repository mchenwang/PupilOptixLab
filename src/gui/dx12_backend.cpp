#include "dx12_backend.h"

#include "static.h"
#include "d3dx12.h"

#include <aclapi.h>
#include <cuda_runtime.h>

#include <comdef.h>
#include <array>
#include <string>
#include <queue>
#include <d3dcompiler.h>
#include <filesystem>

#ifdef _DEBUG
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif

using namespace gui;
using Microsoft::WRL::ComPtr;

extern HWND g_window_handle;
extern uint32_t g_window_w;
extern uint32_t g_window_h;

// static private data
namespace {
bool m_use_warp = false;
ComPtr<IDXGIAdapter4> m_adapter;

bool m_v_sync;
bool m_tearing_supported;
ComPtr<IDXGISwapChain4> m_swapchain;

HANDLE m_fence_event;
uint64_t m_fence_value;
ComPtr<ID3D12Fence> m_fence;
std::array<uint64_t, Backend::NUM_OF_FRAMES> m_frame_fence_values;

struct CmdContext {
    ComPtr<ID3D12CommandAllocator> allocator;
    uint64_t fence_value;
};

std::queue<ComPtr<ID3D12GraphicsCommandList>> m_command_lists;
std::queue<CmdContext> m_context_pool;

uint32_t m_current_index;
std::array<ComPtr<ID3D12Resource>, Backend::NUM_OF_FRAMES> m_back_buffers;
std::array<D3D12_CPU_DESCRIPTOR_HANDLE, Backend::NUM_OF_FRAMES> m_back_buffer_handles;

ComPtr<ID3D12RootSignature> m_root_signature;
ComPtr<ID3D12PipelineState> m_pipeline_state;

ComPtr<ID3D12Resource> m_shared_texture;// screen texture
D3D12_CPU_DESCRIPTOR_HANDLE m_screen_cpu_srv_handle;
D3D12_GPU_DESCRIPTOR_HANDLE m_screen_gpu_srv_handle;

ComPtr<ID3D12Resource> m_vb;
D3D12_VERTEX_BUFFER_VIEW m_vbv;

uint32_t m_cuda_device_id;
uint32_t m_cuda_node_mask;

cudaStream_t m_cuda_stream;
cudaExternalMemory_t m_cuda_ext_memory;
cudaSurfaceObject_t m_cuda_surf_obj;
cudaExternalSemaphore_t m_cuda_semaphore;
}// namespace

namespace {
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_win_security_attributes;
    PSECURITY_DESCRIPTOR m_win_p_security_descriptor;

public:
    WindowsSecurityAttributes();
    ~WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *operator&();
};

inline void StopIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        _com_error err(hr);
        OutputDebugString(err.ErrorMessage());
        assert(false);
    }
}

inline void cudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        std::wstringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";

        OutputDebugString(ss.str().c_str());
        assert(false);
    }
}
#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

void EnableDebugLayer() noexcept;

void InitCuda() noexcept;
void CreateSharedResource(Backend *) noexcept;

void CreateAdapter(Backend *) noexcept;
void CreateDevice(Backend *) noexcept;
void CreateDescHeap(Backend *) noexcept;
void CreateCmdContext(Backend *) noexcept;
void CreateSwapchain(Backend *) noexcept;
void CreatePipeline(Backend *) noexcept;

uint64_t ExecuteCommandLists(Backend *, ComPtr<ID3D12GraphicsCommandList>) noexcept;

[[nodiscard]] ComPtr<ID3D12CommandAllocator> GetCmdAllocator(Backend *) noexcept;

inline void UpdateRenderTarget(Backend *backend) {
    for (uint32_t i = 0; i < Backend::NUM_OF_FRAMES; i++) {
        ComPtr<ID3D12Resource> backbuffer = nullptr;
        m_swapchain->GetBuffer(i, IID_PPV_ARGS(&backbuffer));
        backbuffer->SetName((L"back buffer " + std::to_wstring(i)).data());
        backend->device->CreateRenderTargetView(backbuffer.Get(), nullptr, m_back_buffer_handles[i]);
        m_back_buffers[i] = backbuffer;
    }
}
}// namespace

void Backend::Init() noexcept {
    EnableDebugLayer();
    CreateDevice(this);
    CreateDescHeap(this);
    CreateCmdContext(this);
    CreateSwapchain(this);
    CreatePipeline(this);

    InitCuda();
    CreateSharedResource(this);
}

ComPtr<ID3D12GraphicsCommandList> Backend::GetCmdList() noexcept {
    ComPtr<ID3D12CommandAllocator> allocator = GetCmdAllocator(this);
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

void Backend::Flush() noexcept {
    uint64_t fence_value = ++m_fence_value;
    cmd_queue->Signal(m_fence.Get(), fence_value);

    if (m_fence->GetCompletedValue() < fence_value) {
        m_fence->SetEventOnCompletion(fence_value, m_fence_event);
        ::WaitForSingleObject(m_fence_event, DWORD_MAX);
    }
}

void Backend::RenderScreen(ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    cmd_list->SetGraphicsRootSignature(m_root_signature.Get());
    cmd_list->SetPipelineState(m_pipeline_state.Get());

    ID3D12DescriptorHeap *heaps[] = {srv_heap.Get()};
    cmd_list->SetDescriptorHeaps(1, heaps);
    cmd_list->SetGraphicsRootDescriptorTable(0, m_screen_gpu_srv_handle);// TODO

    D3D12_VIEWPORT viewport{0.f, 0.f, (FLOAT)g_window_w, (FLOAT)g_window_h, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH};
    cmd_list->RSSetViewports(1, &viewport);
    D3D12_RECT rect{0, 0, (LONG)g_window_w, (LONG)g_window_h};
    cmd_list->RSSetScissorRects(1, &rect);

    auto &back_buffer = m_back_buffers[m_current_index];
    auto &back_buffer_rtv = m_back_buffer_handles[m_current_index];
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    cmd_list->OMSetRenderTargets(1, &back_buffer_rtv, TRUE, nullptr);
    const FLOAT clear_color[4]{0.f, 0.f, 0.f, 1.f};
    cmd_list->ClearRenderTargetView(back_buffer_rtv, clear_color, 0, nullptr);

    cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd_list->IASetVertexBuffers(0, 1, &m_vbv);
    cmd_list->DrawInstanced(4, 1, 0, 0);
}

void Backend::Present(ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    auto &back_buffer = m_back_buffers[m_current_index];
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    // execute command
    auto fence_value = ExecuteCommandLists(this, cmd_list);

    m_frame_fence_values[m_current_index] = fence_value;

    uint32_t sync_interval = m_v_sync ? 1 : 0;
    uint32_t present_flags = (m_tearing_supported && !m_v_sync) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    StopIfFailed(m_swapchain->Present(sync_interval, present_flags));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();

    if (m_fence->GetCompletedValue() < m_frame_fence_values[m_current_index]) {
        m_fence->SetEventOnCompletion(m_frame_fence_values[m_current_index], m_fence_event);
        ::WaitForSingleObject(m_fence_event, DWORD_MAX);
    }
}

void Backend::Resize(uint32_t w, uint32_t h) noexcept {
    Flush();
    for (uint32_t i = 0; i < Backend::NUM_OF_FRAMES; i++) {
        m_back_buffers[i].Reset();
    }

    DXGI_SWAP_CHAIN_DESC desc{};
    StopIfFailed(m_swapchain->GetDesc(&desc));
    StopIfFailed(m_swapchain->ResizeBuffers(NUM_OF_FRAMES, w, h, desc.BufferDesc.Format, desc.Flags));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget(this);
}

void Backend::Destroy() noexcept {
    srv_heap.Reset();
    rtv_heap.Reset();
    m_swapchain.Reset();
    m_fence.Reset();
    while (!m_context_pool.empty()) m_context_pool.pop();
    while (!m_command_lists.empty()) m_command_lists.pop();
    for (uint32_t i = 0; i < NUM_OF_FRAMES; i++)
        m_back_buffers[i].Reset();
    m_shared_texture.Reset();
    m_root_signature.Reset();
    m_pipeline_state.Reset();
    m_vb.Reset();
    cmd_queue.Reset();
    device.Reset();
    m_adapter.Reset();
}

namespace {

WindowsSecurityAttributes::WindowsSecurityAttributes() {
    m_win_p_security_descriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    assert(m_win_p_security_descriptor != (PSECURITY_DESCRIPTOR)NULL);

    PSID *ppSID = (PSID *)((PBYTE)m_win_p_security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    InitializeSecurityDescriptor(m_win_p_security_descriptor, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

    SetSecurityDescriptorDacl(m_win_p_security_descriptor, TRUE, *ppACL, FALSE);

    m_win_security_attributes.nLength = sizeof(m_win_security_attributes);
    m_win_security_attributes.lpSecurityDescriptor = m_win_p_security_descriptor;
    m_win_security_attributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID *ppSID = (PSID *)((PBYTE)m_win_p_security_descriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

    if (*ppSID)
        FreeSid(*ppSID);
    if (*ppACL)
        LocalFree(*ppACL);
    free(m_win_p_security_descriptor);
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&() { return &m_win_security_attributes; }

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

        DXGI_INFO_QUEUE_MESSAGE_ID hide[] =
            {
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

void InitCuda() noexcept {
    int num_cuda_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));

    assert(num_cuda_devices);
    DXGI_ADAPTER_DESC1 adapter_desc{};
    m_adapter->GetDesc1(&adapter_desc);

    for (int dev_id = 0; dev_id < num_cuda_devices; dev_id++) {
        cudaDeviceProp dev_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));
        const auto cmp1 =
            memcmp(&adapter_desc.AdapterLuid.LowPart,
                   dev_prop.luid,
                   sizeof(adapter_desc.AdapterLuid.LowPart)) == 0;
        const auto cmp2 =
            memcmp(&adapter_desc.AdapterLuid.HighPart,
                   dev_prop.luid + sizeof(adapter_desc.AdapterLuid.LowPart),
                   sizeof(adapter_desc.AdapterLuid.HighPart)) == 0;

        if (cmp1 && cmp2) {
            CUDA_CHECK(cudaSetDevice(dev_id));
            m_cuda_device_id = (uint32_t)dev_id;
            m_cuda_node_mask = dev_prop.luidDeviceNodeMask;
            CUDA_CHECK(cudaStreamCreate(&m_cuda_stream));
            printf("CUDA Device Used [%d] %s\n", dev_id, dev_prop.name);
            break;
        }
    }
}

void CreateSharedResource(Backend *backend) noexcept {
    auto texture_desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        g_window_w, g_window_h, 1, 0, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
    auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    StopIfFailed(backend->device->CreateCommittedResource(
        &properties, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(&m_shared_texture)));
    m_shared_texture->SetName(L"cuda shared texture");

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    backend->device->CreateShaderResourceView(m_shared_texture.Get(), &srv_desc, m_screen_cpu_srv_handle);

    HANDLE shared_handle{};
    WindowsSecurityAttributes sec_attr{};

    StopIfFailed(backend->device->CreateSharedHandle(m_shared_texture.Get(), &sec_attr, GENERIC_ALL, 0, &shared_handle));
    const auto texAllocInfo = backend->device->GetResourceAllocationInfo(m_cuda_node_mask, 1, &texture_desc);

    cudaExternalMemoryHandleDesc cuda_ext_handle_desc{};
    cuda_ext_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    cuda_ext_handle_desc.handle.win32.handle = shared_handle;
    cuda_ext_handle_desc.size = texAllocInfo.SizeInBytes;
    cuda_ext_handle_desc.flags = cudaExternalMemoryDedicated;
    CUDA_CHECK(cudaImportExternalMemory(&m_cuda_ext_memory, &cuda_ext_handle_desc));

    cudaExternalMemoryMipmappedArrayDesc cuda_ext_mip_desc{};
    cuda_ext_mip_desc.extent = make_cudaExtent(texture_desc.Width, texture_desc.Height, 0);
    cuda_ext_mip_desc.formatDesc = cudaCreateChannelDesc<float4>();
    cuda_ext_mip_desc.numLevels = 1;
    cuda_ext_mip_desc.flags = cudaArraySurfaceLoadStore;

    cudaMipmappedArray_t cuda_mip_array{};
    CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&cuda_mip_array, m_cuda_ext_memory, &cuda_ext_mip_desc));

    cudaArray_t cuda_array{};
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&cuda_array, cuda_mip_array, 0));

    cudaResourceDesc cuda_res_desc{};
    cuda_res_desc.resType = cudaResourceTypeArray;
    cuda_res_desc.res.array.array = cuda_array;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_cuda_surf_obj, &cuda_res_desc));

    CUDA_CHECK(cudaStreamSynchronize(m_cuda_stream));

    // semaphore
    cudaExternalSemaphoreHandleDesc cuda_semaphore_desc{};
    WindowsSecurityAttributes semaphore_sec_attr{};
    LPCWSTR name{};
    HANDLE semaphore_shared_handle{};
    cuda_semaphore_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    backend->device->CreateSharedHandle(m_fence.Get(), &semaphore_sec_attr, GENERIC_ALL, name, &semaphore_shared_handle);
    cuda_semaphore_desc.handle.win32.handle = semaphore_shared_handle;
    cuda_semaphore_desc.flags = 0;
    CUDA_CHECK(cudaImportExternalSemaphore(&m_cuda_semaphore, &cuda_semaphore_desc));
}

ComPtr<ID3D12CommandAllocator> GetCmdAllocator(Backend *backend) noexcept {
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
        StopIfFailed(backend->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));

    return allocator;
}

void CreateDevice(Backend *backend) noexcept {
    if (m_adapter == nullptr) CreateAdapter(backend);
    assert(m_adapter);

    StopIfFailed(D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&backend->device)));
    // Enable debug messages in debug mode.
#ifdef _DEBUG
    ComPtr<ID3D12InfoQueue> info_queue;
    if (SUCCEEDED(backend->device.As(&info_queue))) {
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
        // Suppress whole categories of messages
        //D3D12_MESSAGE_CATEGORY Categories[] = {};

        // Suppress messages based on their severity level
        D3D12_MESSAGE_SEVERITY severities[] = {D3D12_MESSAGE_SEVERITY_INFO};

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

void CreateAdapter(Backend *backend) noexcept {
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

void CreateDescHeap(Backend *backend) noexcept {
    // rtv
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.NumDescriptors = Backend::NUM_OF_FRAMES * 10;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask = 0;
        StopIfFailed(backend->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&backend->rtv_heap)));

        auto size = backend->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        auto rtv_handle = backend->rtv_heap->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t i = 0; i < Backend::NUM_OF_FRAMES; i++) {
            m_back_buffer_handles[i] = rtv_handle;
            rtv_handle.ptr += size;
        }
    }
    // srv
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // one of descripotr for imgui, another one for render result
        desc.NumDescriptors = 2;
        StopIfFailed(backend->device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&backend->srv_heap)));

        auto size = backend->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_screen_gpu_srv_handle = backend->srv_heap->GetGPUDescriptorHandleForHeapStart();
        m_screen_gpu_srv_handle.ptr += size;
        m_screen_cpu_srv_handle = backend->srv_heap->GetCPUDescriptorHandleForHeapStart();
        m_screen_cpu_srv_handle.ptr += size;
    }
}

void CreateCmdContext(Backend *backend) noexcept {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    StopIfFailed(backend->device->CreateCommandQueue(&desc, IID_PPV_ARGS(&backend->cmd_queue)));

    m_fence_value = 0;
    StopIfFailed(backend->device->CreateFence(
        m_fence_value,
        D3D12_FENCE_FLAG_SHARED,// semaphore for cuda
        IID_PPV_ARGS(&m_fence)));
    m_fence_event = ::CreateEvent(NULL, FALSE, FALSE, NULL);
}

void CreateSwapchain(Backend *backend) noexcept {
    m_tearing_supported = CheckTearingSupport();
    ComPtr<IDXGIFactory4> dxgi_factory;
    UINT create_factory_flags = 0;
#if defined(_DEBUG)
    create_factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif

    StopIfFailed(CreateDXGIFactory2(create_factory_flags, IID_PPV_ARGS(&dxgi_factory)));
    DXGI_SWAP_CHAIN_DESC1 swapchain_desc = {};
    swapchain_desc.Width = g_window_w;
    swapchain_desc.Height = g_window_h;
    swapchain_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapchain_desc.Stereo = FALSE;
    swapchain_desc.SampleDesc = {1, 0};
    swapchain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchain_desc.BufferCount = Backend::NUM_OF_FRAMES;
    swapchain_desc.Scaling = DXGI_SCALING_STRETCH;
    swapchain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchain_desc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    // It is recommended to always allow tearing if tearing support is available.
    swapchain_desc.Flags = m_tearing_supported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    ComPtr<IDXGISwapChain1> swapchain;
    StopIfFailed(dxgi_factory->CreateSwapChainForHwnd(
        backend->cmd_queue.Get(),
        g_window_handle,
        &swapchain_desc,
        nullptr,
        nullptr,
        &swapchain));

    StopIfFailed(dxgi_factory->MakeWindowAssociation(g_window_handle, DXGI_MWA_NO_ALT_ENTER));
    StopIfFailed(swapchain.As(&m_swapchain));

    m_current_index = m_swapchain->GetCurrentBackBufferIndex();
    UpdateRenderTarget(backend);
}

void CreatePipeline(Backend *backend) noexcept {
    // root signature
    {
        D3D12_FEATURE_DATA_ROOT_SIGNATURE feat_data{};
        feat_data.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
        if (FAILED(backend->device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feat_data, sizeof(feat_data))))
            assert(false);

        D3D12_DESCRIPTOR_RANGE1 ranges[1]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;// TODO
        ranges[0].NumDescriptors = 1;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;

        CD3DX12_ROOT_PARAMETER1 root_params[1]{};
        root_params[0].InitAsDescriptorTable(1, ranges, D3D12_SHADER_VISIBILITY_PIXEL);

        D3D12_STATIC_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.RegisterSpace = 0;
        sampler_desc.ShaderRegister = 0;
        sampler_desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        sampler_desc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        sampler_desc.MaxAnisotropy = 16;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        sampler_desc.MinLOD = 0.f;
        sampler_desc.MipLODBias = 0.f;

        D3D12_ROOT_SIGNATURE_FLAGS root_sign_flags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_sign_desc{};
        root_sign_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        root_sign_desc.Desc_1_1.Flags = root_sign_flags;
        root_sign_desc.Desc_1_1.NumParameters = 1;
        root_sign_desc.Desc_1_1.NumStaticSamplers = 1;
        root_sign_desc.Desc_1_1.pParameters = root_params;// TODO
        root_sign_desc.Desc_1_1.pStaticSamplers = &sampler_desc;

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        StopIfFailed(D3D12SerializeVersionedRootSignature(&root_sign_desc, signature.GetAddressOf(), error.GetAddressOf()));
        StopIfFailed(backend->device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_root_signature)));
    }

    // pso
    {
        ComPtr<ID3DBlob> vs;
        ComPtr<ID3DBlob> ps;

#if defined(_DEBUG)
        // Enable better shader debugging with the graphics debugging tools.
        UINT compile_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
        UINT compile_flags = 0;
#endif

        std::filesystem::path file_path = (std::filesystem::path{CODE_DIR} / "gui/shader.hlsl").make_preferred();
        std::wstring w_file_path = file_path.wstring();
        LPCWSTR result = w_file_path.data();
        //StopIfFailed(D3DCompileFromFile(result, 0, 0, "VSMain", "vs_5_1", compile_flags, 0, &vs, 0));
        //StopIfFailed(D3DCompileFromFile(result, 0, 0, "PSMain", "ps_5_1", compile_flags, 0, &ps, 0));
        ComPtr<ID3DBlob> errors;
        auto hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "VSMain", "vs_5_1", compile_flags, 0, &vs, &errors);
        if (errors != nullptr)
            OutputDebugStringA((char *)errors->GetBufferPointer());
        StopIfFailed(hr);
        hr = D3DCompileFromFile(result, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "PSMain", "ps_5_1", compile_flags, 0, &ps, &errors);
        if (errors != nullptr)
            OutputDebugStringA((char *)errors->GetBufferPointer());
        StopIfFailed(hr);

        // Define the vertex input layout.
        D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
            {{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
             {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};

        // Describe and create the graphics pipeline state object (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.InputLayout = {inputElementDescs, _countof(inputElementDescs)};
        pso_desc.pRootSignature = m_root_signature.Get();
        pso_desc.VS.BytecodeLength = vs->GetBufferSize();
        pso_desc.VS.pShaderBytecode = vs->GetBufferPointer();
        pso_desc.PS.BytecodeLength = ps->GetBufferSize();
        pso_desc.PS.pShaderBytecode = ps->GetBufferPointer();
        pso_desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        pso_desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        pso_desc.DepthStencilState.DepthEnable = FALSE;
        pso_desc.DepthStencilState.StencilEnable = FALSE;
        pso_desc.SampleMask = UINT_MAX;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        StopIfFailed(backend->device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&m_pipeline_state)));
    }

    // upload vb
    auto cmd_list = backend->GetCmdList();
    ComPtr<ID3D12Resource> vb_upload = nullptr;
    {
        struct TriVertex {
            float x, y, z;
            float u, v;
            TriVertex(float x, float y, float z, float u, float v) noexcept
                : x(x), y(y), z(z), u(u), v(v) {}
        };
        TriVertex quad[] = {
            {-1.f, -1.f, 0.f, 0.f, 0.f},
            {-1.f, 1.f, 0.f, 0.f, 1.f},
            {1.f, -1.f, 0.f, 1.f, 0.f},
            {1.f, 1.f, 0.f, 1.f, 1.f}};

        constexpr auto vb_size = sizeof(quad);
        D3D12_HEAP_PROPERTIES heap_properties{};
        heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_properties.CreationNodeMask = 1;
        heap_properties.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC vb_desc{};
        vb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        vb_desc.Alignment = 0;
        vb_desc.Width = vb_size;
        vb_desc.Height = 1;
        vb_desc.DepthOrArraySize = 1;
        vb_desc.MipLevels = 1;
        vb_desc.Format = DXGI_FORMAT_UNKNOWN;
        vb_desc.SampleDesc.Count = 1;
        vb_desc.SampleDesc.Quality = 0;
        vb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        vb_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        StopIfFailed(backend->device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &vb_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&m_vb)));

        heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
        StopIfFailed(backend->device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &vb_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&vb_upload)));

        D3D12_SUBRESOURCE_DATA vertex_data{};
        vertex_data.pData = quad;
        vertex_data.RowPitch = vb_size;
        vertex_data.SlicePitch = vertex_data.RowPitch;

        UpdateSubresources<1>(cmd_list.Get(), m_vb.Get(), vb_upload.Get(), 0, 0, 1, &vertex_data);
        D3D12_RESOURCE_BARRIER barrier =
            CD3DX12_RESOURCE_BARRIER::Transition(
                m_vb.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        cmd_list->ResourceBarrier(1, &barrier);

        // Initialize the vertex buffer view.
        m_vbv.BufferLocation = m_vb->GetGPUVirtualAddress();
        m_vbv.StrideInBytes = sizeof(TriVertex);
        m_vbv.SizeInBytes = vb_size;
    }

    ExecuteCommandLists(backend, cmd_list);
    backend->Flush();
}

uint64_t ExecuteCommandLists(Backend *backend, ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {

    cmd_list->Close();

    ID3D12CommandAllocator *allocator;
    UINT dataSize = sizeof(allocator);
    StopIfFailed(cmd_list->GetPrivateData(__uuidof(ID3D12CommandAllocator), &dataSize, &allocator));

    ID3D12CommandList *const p_cmd_lists[] = {cmd_list.Get()};
    backend->cmd_queue->ExecuteCommandLists(1, p_cmd_lists);

    uint64_t fence_value = ++m_fence_value;
    backend->cmd_queue->Signal(m_fence.Get(), fence_value);
    m_context_pool.emplace(allocator, fence_value);
    m_command_lists.emplace(cmd_list);
    allocator->Release();

    return fence_value;
}
}// namespace
