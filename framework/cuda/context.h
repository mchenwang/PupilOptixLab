#pragma once

#include "util/util.h"
#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <queue>

namespace Pupil::DirectX {
class Context;
};

namespace Pupil::cuda {
class Stream;
class Context : public Pupil::util::Singleton<Context> {
public:
    CUcontext context = nullptr;
    uint32_t cuda_device_id = 0;
    uint32_t cuda_node_mask = 0;

    operator CUcontext() const noexcept { return context; }

    void Init() noexcept;
    void Destroy() noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }
    void Synchronize() noexcept { CUDA_CHECK(cudaDeviceSynchronize()); }

private:
    bool m_init_flag = false;
};
}// namespace Pupil::cuda