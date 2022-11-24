#include "optix_device.h"
#include "optix_device.h"
#include "wsa.h"
#include "d3dx12.h"

#include "common/util.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <iostream>
#include <format>

using namespace device;
using Microsoft::WRL::ComPtr;

namespace {
void ContextLogCB(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << std::format("[{:2}][{:12}]: {}\n", level, tag, message);
}
}// namespace

Optix::Optix(DX12 *dx12_backend) noexcept {
    m_dx12_backend = dx12_backend;

    InitCuda();
    CUcontext cu_ctx = 0;
    CUDA_CHECK(cuCtxGetCurrent(&cu_ctx));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options{};
    options.logCallbackFunction = &ContextLogCB;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));
}

void Optix::InitCuda() noexcept {
    int num_cuda_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_cuda_devices));
    assert(num_cuda_devices);

    DXGI_ADAPTER_DESC1 dxgi_adapter_desc{};
    m_dx12_backend->m_adapter->GetDesc1(&dxgi_adapter_desc);

    for (int dev_id = 0; dev_id < num_cuda_devices; dev_id++) {
        cudaDeviceProp dev_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));
        const auto cmp1 =
            memcmp(&dxgi_adapter_desc.AdapterLuid.LowPart,
                   dev_prop.luid,
                   sizeof(dxgi_adapter_desc.AdapterLuid.LowPart)) == 0;
        const auto cmp2 =
            memcmp(&dxgi_adapter_desc.AdapterLuid.HighPart,
                   dev_prop.luid + sizeof(dxgi_adapter_desc.AdapterLuid.LowPart),
                   sizeof(dxgi_adapter_desc.AdapterLuid.HighPart)) == 0;

        if (cmp1 && cmp2) {
            CUDA_CHECK(cudaSetDevice(dev_id));
            cuda_device_id = (uint32_t)dev_id;
            cuda_node_mask = dev_prop.luidDeviceNodeMask;
            CUDA_CHECK(cudaStreamCreate(&cuda_stream));
            std::cout << std::format("CUDA Device Used [{}] {}\n", dev_id, dev_prop.name);
            break;
        }
    }

    CUDA_CHECK(cudaFree(0));

    // semaphore
    cudaExternalSemaphoreHandleDesc cuda_semaphore_desc{};
    WindowsSecurityAttributes semaphore_sec_attr{};
    LPCWSTR name = L"Cuda semaphore";
    HANDLE semaphore_shared_handle{};
    cuda_semaphore_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    m_dx12_backend->device->CreateSharedHandle(m_dx12_backend->m_fence.Get(), &semaphore_sec_attr, GENERIC_ALL, name, &semaphore_shared_handle);
    cuda_semaphore_desc.handle.win32.handle = semaphore_shared_handle;
    cuda_semaphore_desc.flags = 0;
    CUDA_CHECK(cudaImportExternalSemaphore(&cuda_semaphore, &cuda_semaphore_desc));
    CloseHandle(semaphore_shared_handle);
}

[[nodiscard]] std::unique_ptr<CudaDx12SharedTexture> Optix::CreateSharedResourceWithDX12() noexcept {
    auto target = std::make_unique<CudaDx12SharedTexture>();

    auto texture_desc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32G32B32A32_FLOAT,
        m_dx12_backend->m_frame_w,
        m_dx12_backend->m_frame_h,
        1, 0, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
    auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    StopIfFailed(m_dx12_backend->device->CreateCommittedResource(
        &properties, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr, IID_PPV_ARGS(&target->dx12_resource)));
    target->dx12_resource->SetName(L"cuda shared texture");

    HANDLE shared_handle{};
    WindowsSecurityAttributes sec_attr{};

    StopIfFailed(m_dx12_backend->device->CreateSharedHandle(target->dx12_resource.Get(), &sec_attr, GENERIC_ALL, 0, &shared_handle));
    const auto texAllocInfo = m_dx12_backend->device->GetResourceAllocationInfo(cuda_node_mask, 1, &texture_desc);

    cudaExternalMemoryHandleDesc cuda_ext_handle_desc{};
    cuda_ext_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    cuda_ext_handle_desc.handle.win32.handle = shared_handle;
    cuda_ext_handle_desc.size = texAllocInfo.SizeInBytes;
    cuda_ext_handle_desc.flags = cudaExternalMemoryDedicated;
    CUDA_CHECK(cudaImportExternalMemory(&target->cuda_ext_memory, &cuda_ext_handle_desc));

    cudaExternalMemoryMipmappedArrayDesc cuda_ext_mip_desc{};
    cuda_ext_mip_desc.extent = make_cudaExtent(texture_desc.Width, texture_desc.Height, 0);
    cuda_ext_mip_desc.formatDesc = cudaCreateChannelDesc<float4>();
    cuda_ext_mip_desc.numLevels = 1;
    cuda_ext_mip_desc.flags = cudaArraySurfaceLoadStore;

    cudaMipmappedArray_t cuda_mip_array{};
    CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&cuda_mip_array, target->cuda_ext_memory, &cuda_ext_mip_desc));

    cudaArray_t cuda_array{};
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&cuda_array, cuda_mip_array, 0));

    cudaResourceDesc cuda_res_desc{};
    cuda_res_desc.resType = cudaResourceTypeArray;
    cuda_res_desc.res.array.array = cuda_array;
    CUDA_CHECK(cudaCreateSurfaceObject(&target->cuda_surf_obj, &cuda_res_desc));

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    return std::move(target);
}

std::unique_ptr<SharedFrameResource> device::Optix::CreateSharedFrameResource() noexcept {
    auto target = std::make_unique<SharedFrameResource>();
    for (auto i = 0u; i < DX12::NUM_OF_FRAMES; i++) {
        target->frame[i] = CreateSharedResourceWithDX12();
    }
    m_frame_resource = target.get();
    return std::move(target);
}

void Optix::Run() noexcept {
    uint32_t frame_index = m_dx12_backend->GetCurrentFrameIndex();
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.params.fence.value = m_frame_resource->frame[frame_index]->fence_value;
    cudaWaitExternalSemaphoresAsync(&cuda_semaphore, &wait_params, 1, cuda_stream);

    // do ray tracing

    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.params.fence.value = m_frame_resource->frame[frame_index]->fence_value + 1;
    cudaSignalExternalSemaphoresAsync(&cuda_semaphore, &signal_params, 1, cuda_stream);
}

void Optix::InitPipeline(const optix_wrap::PipelineDesc &desc) noexcept {
    pipeline = std::make_unique<optix_wrap::Pipeline>(this, desc);
}

void Optix::InitScene() noexcept {
}
