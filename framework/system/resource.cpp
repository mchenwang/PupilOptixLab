#include "resource.h"

#include "cuda/util.h"
#include "cuda/context.h"
#include "dx12/d3dx12.h"
#include "dx12/context.h"

#include "wsa.h"

#include <iostream>

namespace Pupil {
Buffer::~Buffer() noexcept {
    switch (type) {
        case EBufferType::Cuda: {
            CUDA_FREE(cuda_res.ptr);
        } break;
        case EBufferType::SharedCudaWithDX12: {
            shared_res.dx12_ptr = nullptr;
            if (shared_res.cuda_ext_memory)
                CUDA_CHECK(cudaDestroyExternalMemory(shared_res.cuda_ext_memory));
            shared_res.cuda_ext_memory = 0;
            shared_res.cuda_ptr = 0;
        } break;
        case EBufferType::DX12: {
            dx12_res.ptr = nullptr;
        } break;
    }
}

Buffer *BufferManager::GetBuffer(std::string_view id) noexcept {
    auto it = m_buffers.find(id);
    return it == m_buffers.end() ? nullptr : it->second.get();
}

Buffer *BufferManager::AllocBuffer(const BufferDesc &desc) noexcept {
    auto buffer = std::make_unique<Buffer>();

    buffer->type = desc.type;
    switch (desc.type) {
        case EBufferType::Cuda: {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&buffer->cuda_res.ptr),
                desc.size));
        } break;
        case EBufferType::SharedCudaWithDX12: {
            auto d3d12_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(
                desc.size,
                D3D12_RESOURCE_FLAG_NONE);
            auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

            auto dx12_context = util::Singleton<DirectX::Context>::instance();
            DirectX::StopIfFailed(dx12_context->device->CreateCommittedResource(
                &properties, D3D12_HEAP_FLAG_SHARED, &d3d12_buffer_desc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
                winrt::guid_of<ID3D12Resource>(), buffer->shared_res.dx12_ptr.put_void()));
            buffer->shared_res.dx12_ptr->SetName(std::wstring{ desc.name.begin(), desc.name.end() }.c_str());

            HANDLE shared_handle{};
            WindowsSecurityAttributes sec_attr{};

            DirectX::StopIfFailed(dx12_context->device->CreateSharedHandle(buffer->shared_res.dx12_ptr.get(), &sec_attr, GENERIC_ALL, 0, &shared_handle));

            auto cuda_contex = util::Singleton<cuda::Context>::instance();
            const auto texAllocInfo = dx12_context->device->GetResourceAllocationInfo(cuda_contex->cuda_node_mask, 1, &d3d12_buffer_desc);

            cudaExternalMemoryHandleDesc cuda_ext_handle_desc{};
            cuda_ext_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
            cuda_ext_handle_desc.handle.win32.handle = shared_handle;
            cuda_ext_handle_desc.size = texAllocInfo.SizeInBytes;
            cuda_ext_handle_desc.flags = cudaExternalMemoryDedicated;
            CUDA_CHECK(cudaImportExternalMemory(&buffer->shared_res.cuda_ext_memory, &cuda_ext_handle_desc));
            CloseHandle(shared_handle);

            cudaExternalMemoryBufferDesc cuda_ext_buffer_desc{};
            cuda_ext_buffer_desc.offset = 0;
            cuda_ext_buffer_desc.size = desc.size;
            CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(
                reinterpret_cast<void **>(&buffer->shared_res.cuda_ptr),
                buffer->shared_res.cuda_ext_memory, &cuda_ext_buffer_desc));

            // CUDA_CHECK(cudaStreamSynchronize(cuda_contex->));
        } break;
        case EBufferType::DX12: {
            assert(false);
        } break;
    }

    auto ret = buffer.get();
    auto it = m_buffers.find(desc.name);
    if (it != m_buffers.end()) {
        printf("warning: buffer[%s] is reset.\n", desc.name.c_str());
    }
    m_buffers[desc.name] = std::move(buffer);
    return ret;
}

void BufferManager::AddBuffer(std::string_view id, std::unique_ptr<Buffer> &buffer) noexcept {
    auto it = m_buffers.find(id);
    if (it != m_buffers.end()) {
        printf("warning: buffer[%s] is reset.\n", id.data());
    }
    m_buffers[std::string{ id }] = std::move(buffer);
}
}// namespace Pupil