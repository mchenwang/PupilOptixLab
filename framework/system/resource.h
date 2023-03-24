#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <d3d12.h>
#include <winrt/base.h>

#include "util/util.h"

namespace Pupil {
enum class EBufferType {
    Cuda,
    DX12,
    SharedCudaWithDX12
};

struct BufferDesc {
    EBufferType type;
    std::string name;
    uint64_t size;
};

struct CudaBuffer {
    CUdeviceptr ptr = 0;
};

struct DX12Buffer {
    winrt::com_ptr<ID3D12Resource> ptr = nullptr;
};

struct SharedBuffer {
    winrt::com_ptr<ID3D12Resource> dx12_ptr = nullptr;
    CUdeviceptr cuda_ptr = 0;
    cudaExternalMemory_t cuda_ext_memory = nullptr;
};

struct Buffer {
    EBufferType type;
    union {
        CudaBuffer cuda_res;
        DX12Buffer dx12_res;
        SharedBuffer shared_res;
    };

    Buffer() noexcept {}
    ~Buffer() noexcept;
};

class BufferManager : public util::Singleton<BufferManager> {
public:
    [[nodiscard]] Buffer *GetBuffer(std::string_view) noexcept;
    Buffer *AllocBuffer(const BufferDesc &) noexcept;
    void AddBuffer(std::string_view id, std::unique_ptr<Buffer> &) noexcept;

private:
    std::unordered_map<std::string, std::unique_ptr<Buffer>, util::StringHash, std::equal_to<>> m_buffers;
};
}// namespace Pupil