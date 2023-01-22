#include "dx12_backend.h"
#include "device/d3dx12.h"
#include "device/optix_device.h"

#include "static.h"

#include <array>
#include <string>
#include <queue>
#include <d3dcompiler.h>
#include <filesystem>

using namespace gui;
using Microsoft::WRL::ComPtr;

extern HWND g_window_handle;
extern uint32_t g_window_w;
extern uint32_t g_window_h;

// static private data
namespace {
ComPtr<ID3D12RootSignature> m_root_signature;
ComPtr<ID3D12PipelineState> m_pipeline_state;

ComPtr<ID3D12Resource> m_vb;
D3D12_VERTEX_BUFFER_VIEW m_vbv;

ComPtr<ID3D12Resource> m_frame_constant_buffer;
void *m_frame_constant_buffer_mapped_ptr = nullptr;
}// namespace

namespace {
void CreatePipeline(device::DX12 *) noexcept;
void CreateResource(device::DX12 *) noexcept;
}// namespace

void Backend::Init() noexcept {
    m_backend = std::make_unique<device::DX12>(g_window_w, g_window_h, g_window_handle);

    CreatePipeline(m_backend.get());
    CreateResource(m_backend.get());
}

void Backend::Resize(uint32_t w, uint32_t h) noexcept {
    m_frame_info.w = w;
    m_frame_info.h = h;

    gui::FramInfo init_frame_info{
        .w = w,
        .h = h
    };
    memcpy(m_frame_constant_buffer_mapped_ptr, &init_frame_info, sizeof(gui::FramInfo));

    m_backend->Resize(w, h);
}

void Backend::SetScreenResource(device::SharedFrameResource *shared_frame_resource) noexcept {
    auto size = m_backend->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu_srv_handle = m_backend->srv_heap->GetCPUDescriptorHandleForHeapStart();
    auto gpu_srv_handle = m_backend->srv_heap->GetGPUDescriptorHandleForHeapStart();

    for (auto i = 0u; i < device::DX12::NUM_OF_FRAMES; i++) {
        frames[i].src = shared_frame_resource->frame[i].get();
        frames[i].src->fence_value = m_backend->GetGlobalFenceValue();
        frames[i].screen_texture = shared_frame_resource->frame[i]->dx12_resource;

        gpu_srv_handle.ptr += size;
        frames[i].screen_gpu_srv_handle = gpu_srv_handle;

        cpu_srv_handle.ptr += size;
        frames[i].screen_cpu_srv_handle = cpu_srv_handle;

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.Buffer.NumElements = g_window_h * g_window_w;
        srv_desc.Buffer.StructureByteStride = sizeof(float) * 4;
        srv_desc.Buffer.FirstElement = 0;
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        /*srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;*/
        m_backend->device->CreateShaderResourceView(frames[i].screen_texture.Get(), &srv_desc, frames[i].screen_cpu_srv_handle);
    }
}

void Backend::RenderScreen(ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    cmd_list->SetGraphicsRootSignature(m_root_signature.Get());
    cmd_list->SetPipelineState(m_pipeline_state.Get());

    auto &frame = frames[m_backend->GetCurrentFrameIndex()];
    ID3D12DescriptorHeap *heaps[] = { m_backend->srv_heap.Get() };
    cmd_list->SetDescriptorHeaps(1, heaps);
    cmd_list->SetGraphicsRootDescriptorTable(0, frame.screen_gpu_srv_handle);
    cmd_list->SetGraphicsRootConstantBufferView(1, m_frame_constant_buffer->GetGPUVirtualAddress());

    D3D12_VIEWPORT viewport{ 0.f, 0.f, (FLOAT)g_window_w, (FLOAT)g_window_h, D3D12_MIN_DEPTH, D3D12_MAX_DEPTH };
    cmd_list->RSSetViewports(1, &viewport);
    D3D12_RECT rect{ 0, 0, (LONG)g_window_w, (LONG)g_window_h };
    cmd_list->RSSetScissorRects(1, &rect);

    auto [back_buffer, back_buffer_rtv] = m_backend->GetCurrentFrame();
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = back_buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    cmd_list->OMSetRenderTargets(1, &back_buffer_rtv, TRUE, nullptr);
    const FLOAT clear_color[4]{ 0.f, 0.f, 0.f, 1.f };
    cmd_list->ClearRenderTargetView(back_buffer_rtv, clear_color, 0, nullptr);

    cmd_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cmd_list->IASetVertexBuffers(0, 1, &m_vbv);
    cmd_list->DrawInstanced(4, 1, 0, 0);
}

void Backend::Present(ComPtr<ID3D12GraphicsCommandList> cmd_list) noexcept {
    auto fence_value = m_backend->Present(cmd_list);
    auto &frame = frames[m_backend->GetCurrentFrameIndex()];
    frame.src->fence_value = fence_value;
    m_backend->global_fence_value = fence_value + 1;
    m_backend->SetCurrentFrameFenceValue(fence_value + 1);
    m_backend->MoveToNextFrame();
}

void Backend::Destroy() noexcept {
    if (m_frame_constant_buffer_mapped_ptr) {
        m_frame_constant_buffer->Unmap(0, nullptr);
        m_frame_constant_buffer_mapped_ptr = nullptr;
    }
    m_root_signature.Reset();
    m_pipeline_state.Reset();
    m_vb.Reset();
    m_backend.reset();
}

namespace {
void CreatePipeline(device::DX12 *backend) noexcept {
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

        CD3DX12_ROOT_PARAMETER1 root_params[2]{};
        root_params[0].InitAsDescriptorTable(1, ranges, D3D12_SHADER_VISIBILITY_PIXEL);
        root_params[1].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC, D3D12_SHADER_VISIBILITY_PIXEL);

        D3D12_ROOT_SIGNATURE_FLAGS root_sign_flags =
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_sign_desc{};
        root_sign_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        root_sign_desc.Desc_1_1.Flags = root_sign_flags;
        root_sign_desc.Desc_1_1.NumParameters = 2;
        root_sign_desc.Desc_1_1.NumStaticSamplers = 0;
        root_sign_desc.Desc_1_1.pParameters = root_params;// TODO
        // root_sign_desc.Desc_1_1.pStaticSamplers = &sampler_desc;

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

        std::filesystem::path file_path = (std::filesystem::path{ CODE_DIR } / "gui/shader.hlsl").make_preferred();
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
        D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
        };

        // Describe and create the graphics pipeline state object (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
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
            { -1.f, -1.f, 0.f, 0.f, 0.f },
            { -1.f, 1.f, 0.f, 0.f, 1.f },
            { 1.f, -1.f, 0.f, 1.f, 0.f },
            { 1.f, 1.f, 0.f, 1.f, 1.f }
        };

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

    backend->ExecuteCommandLists(cmd_list);
    backend->Flush();
}

void CreateResource(device::DX12 *backend) noexcept {
    // for frame info
    CD3DX12_HEAP_PROPERTIES properties(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(gui::FramInfo));
    StopIfFailed(backend->device->CreateCommittedResource(
        &properties,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_frame_constant_buffer)));

    m_frame_constant_buffer->Map(0, nullptr, &m_frame_constant_buffer_mapped_ptr);

    gui::FramInfo init_frame_info{
        .w = g_window_w,
        .h = g_window_h
    };
    memcpy(m_frame_constant_buffer_mapped_ptr, &init_frame_info, sizeof(gui::FramInfo));
}
}// namespace
