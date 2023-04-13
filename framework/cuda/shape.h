#pragma once

#include "preprocessor.h"
#include "cuda/vec_math.h"
#include <cuda.h>
#include <unordered_map>
#include "util/util.h"

namespace Pupil::cuda {
class CudaShapeDataManager : public util::Singleton<CudaShapeDataManager> {
private:
    std::unordered_map<void *, CUdeviceptr> m_cuda_geometry_map;

public:
    [[nodiscard]] CUdeviceptr GetCudaMemPtr(void *p_data, size_t size) noexcept;

    void Clear() noexcept;
};
}// namespace Pupil::cuda