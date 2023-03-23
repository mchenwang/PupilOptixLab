#pragma once

#include "util/util.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_map>

namespace Pupil::DirectX {
class Context;
};

namespace Pupil::cuda {
class Context : Pupil::util::Singleton<Context> {
public:
    constexpr static std::string_view DEFAULT_STREAM = "_default";
    constexpr static std::string_view ASYN_STREAM = "_asyn";

    CUcontext context = nullptr;
    uint32_t cuda_device_id = 0;
    uint32_t cuda_node_mask = 0;

    operator CUcontext() const noexcept { return context; }

    void Init() noexcept;
    void Init(DirectX::Context *) noexcept;
    void Destroy() noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }
    [[nodiscard]] cudaStream_t GetStream(std::string_view) noexcept;

private:
    std::unordered_map<std::string, cudaStream_t, util::StringHash, std::equal_to<>> m_streams;
    bool m_init_flag = false;
};
}// namespace Pupil::cuda