#pragma once

#include "common/util.h"
#include "cuda_util/texture.h"

namespace util {
struct Texture;
}

namespace device {
class CudaTextureManager : public util::Singleton<CudaTextureManager> {
private:
    std::vector<cudaArray_t> m_cuda_memory_array;

public:
    [[nodiscard]] cudaTextureObject_t GetCudaTextureObject(util::Texture) noexcept;
    [[nodiscard]] cuda::Texture GetCudaTexture(util::Texture) noexcept;

    void Clear() noexcept;
};
}// namespace device