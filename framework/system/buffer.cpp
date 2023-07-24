#include "buffer.h"
#include "cuda/util.h"
#include "cuda/context.h"
#include "dx12/d3dx12.h"
#include "dx12/context.h"
#include "util/log.h"

#include "wsa.h"

namespace Pupil {
Buffer::~Buffer() noexcept {
    CUDA_FREE(cuda_ptr);
    dx12_ptr = nullptr;
}

BufferManager::BufferManager() noexcept {
    m_buffer_names.emplace_back(DEFAULT_FINAL_RESULT_BUFFER_NAME);
}

BufferManager::~BufferManager() noexcept {
    Destroy();
}

void BufferManager::Destroy() noexcept {
    CUDA_SYNC_CHECK();
    m_buffers.clear();
    for (auto &&[dx12_ptr, cuda_ext_memory] : m_cuda_ext_memorys) {
        if (cuda_ext_memory)
            CUDA_CHECK(cudaDestroyExternalMemory(cuda_ext_memory));
    }
    m_cuda_ext_memorys.clear();
}

Buffer *BufferManager::GetBuffer(std::string_view id) noexcept {
    auto it = m_buffers.find(id);
    return it == m_buffers.end() ? nullptr : it->second.get();
}

Buffer *BufferManager::AllocBuffer(const BufferDesc &desc) noexcept {
    auto buffer = std::make_unique<Buffer>(desc);

    auto size = static_cast<size_t>(desc.width * desc.height * desc.stride_in_byte);
    if (!(desc.flag & EBufferFlag::SharedWithDX12)) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer->cuda_ptr), size));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(buffer->cuda_ptr), 0, size));
    } else {
        auto d3d12_buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_NONE);
        auto properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

        auto dx12_context = util::Singleton<DirectX::Context>::instance();
        winrt::com_ptr<ID3D12Resource> temp_res;
        DirectX::StopIfFailed(dx12_context->device->CreateCommittedResource(
            &properties, D3D12_HEAP_FLAG_SHARED, &d3d12_buffer_desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
            winrt::guid_of<ID3D12Resource>(), temp_res.put_void()));
        buffer->dx12_ptr = temp_res;
        if (desc.name != nullptr)
            buffer->dx12_ptr->SetName(std::wstring{ desc.name, desc.name + strlen(desc.name) }.c_str());

        HANDLE shared_handle{};
        WindowsSecurityAttributes sec_attr{};

        DirectX::StopIfFailed(dx12_context->device->CreateSharedHandle(buffer->dx12_ptr.get(), &sec_attr, GENERIC_ALL, 0, &shared_handle));

        auto cuda_contex = util::Singleton<cuda::Context>::instance();
        const auto texAllocInfo = dx12_context->device->GetResourceAllocationInfo(cuda_contex->cuda_node_mask, 1, &d3d12_buffer_desc);

        cudaExternalMemory_t cuda_ext_memory;
        cudaExternalMemoryHandleDesc cuda_ext_handle_desc{};
        cuda_ext_handle_desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
        cuda_ext_handle_desc.handle.win32.handle = shared_handle;
        cuda_ext_handle_desc.size = texAllocInfo.SizeInBytes;
        cuda_ext_handle_desc.flags = cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&cuda_ext_memory, &cuda_ext_handle_desc));
        CloseHandle(shared_handle);

        cudaExternalMemoryBufferDesc cuda_ext_buffer_desc{};
        cuda_ext_buffer_desc.offset = 0;
        cuda_ext_buffer_desc.size = size;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(
            reinterpret_cast<void **>(&buffer->cuda_ptr),
            cuda_ext_memory, &cuda_ext_buffer_desc));

        m_cuda_ext_memorys[buffer->dx12_ptr.get()] = cuda_ext_memory;
    }

    auto ret = buffer.get();
    if (m_buffers.find(desc.name) == m_buffers.end()) {
        if (desc.name != DEFAULT_FINAL_RESULT_BUFFER_NAME) {
            m_buffer_names.emplace_back(desc.name);
        }
    }
    // else {
    //     Pupil::Log::Warn("buffer[{}] is reset.", desc.name);
    // }
    m_buffers[desc.name] = std::move(buffer);

    return ret;
}

void BufferManager::AddBuffer(std::string_view id, std::unique_ptr<Buffer> &buffer) noexcept {
    if (m_buffers.find(id) == m_buffers.end()) {
        if (id != DEFAULT_FINAL_RESULT_BUFFER_NAME) {
            m_buffer_names.emplace_back(id);
        }
    }
    // else {
    //     Pupil::Log::Warn("buffer[{}] is reset.", id);
    // }

    m_buffers[id.data()] = std::move(buffer);
}
}// namespace Pupil