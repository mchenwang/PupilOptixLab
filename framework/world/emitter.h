#pragma once

#include "render/emitter.h"

namespace Pupil::world {
class EmitterHelper {
private:
    std::vector<optix::Emitter> m_areas;
    std::vector<optix::Emitter> m_points;
    std::vector<optix::Emitter> m_directionals;
    optix::Emitter m_env;

    CUdeviceptr m_areas_cuda_memory;
    CUdeviceptr m_points_cuda_memory;
    CUdeviceptr m_directionals_cuda_memory;
    CUdeviceptr m_env_cuda_memory;
    CUdeviceptr m_env_cdf_weight_cuda_memory;

    void GenerateEmitters(resource::Scene *) noexcept;

public:
    EmitterHelper(resource::Scene *) noexcept;
    ~EmitterHelper() noexcept;

    void Clear() noexcept;
    void Reset(resource::Scene *) noexcept;
    optix::EmitterGroup GetEmitterGroup() noexcept;
};
}// namespace Pupil::world