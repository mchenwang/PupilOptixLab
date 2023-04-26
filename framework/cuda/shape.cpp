#include "shape.h"
#include "util.h"

namespace Pupil::cuda {
CUdeviceptr CudaShapeDataManager::GetCudaMemPtr(void *p_data, size_t size) noexcept {
    if (p_data == nullptr || size == 0) return 0;
    auto it = m_cuda_geometry_map.find(p_data);
    if (it == m_cuda_geometry_map.end()) {
        m_cuda_geometry_map[p_data] = Pupil::cuda::CudaMemcpyToDevice(p_data, size);
    }
    return m_cuda_geometry_map[p_data];
}

void CudaShapeDataManager::Clear() noexcept {
    m_cuda_geometry_map.clear();
}
}// namespace Pupil::cuda