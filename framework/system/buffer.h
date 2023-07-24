#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <d3d12.h>
#include <winrt/base.h>

#include "util/util.h"

namespace Pupil {

enum class EBufferFlag : unsigned int {
    None = 0,
    SharedWithDX12 = 1,
    AllowDisplay = 1 << 1,
};

inline bool operator&(const EBufferFlag &target, const EBufferFlag &flag) noexcept {
    return (static_cast<unsigned int>(target) & static_cast<unsigned int>(flag)) ==
           static_cast<unsigned int>(flag);
}

struct BufferDesc {
    const char *name = nullptr;
    EBufferFlag flag = EBufferFlag::None;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t stride_in_byte = 1;
};

struct Buffer {
    BufferDesc desc{};
    CUdeviceptr cuda_ptr = 0;
    winrt::com_ptr<ID3D12Resource> dx12_ptr = nullptr;

    Buffer() noexcept = default;
    Buffer(const BufferDesc &desc) noexcept : desc(desc) {}
    ~Buffer() noexcept;
};

class BufferManager : public util::Singleton<BufferManager> {
public:
    constexpr static std::string_view DEFAULT_FINAL_RESULT_BUFFER_NAME = "final result";

    BufferManager() noexcept;
    ~BufferManager() noexcept;

    void Destroy() noexcept;

    [[nodiscard]] Buffer *GetBuffer(std::string_view) noexcept;
    Buffer *AllocBuffer(const BufferDesc &) noexcept;
    void AddBuffer(std::string_view id, std::unique_ptr<Buffer> &) noexcept;

    [[nodiscard]] const std::vector<std::string> &GetBufferNameList() const noexcept { return m_buffer_names; }

private:
    std::unordered_map<std::string, std::unique_ptr<Buffer>, util::StringHash, std::equal_to<>> m_buffers;
    std::unordered_map<ID3D12Resource *, cudaExternalMemory_t> m_cuda_ext_memorys;
    std::vector<std::string> m_buffer_names;
};
}// namespace Pupil