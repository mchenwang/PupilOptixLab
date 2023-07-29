#pragma once

#include "render/emitter.h"
#include "resource/emitter.h"
#include "resource/shape.h"

namespace Pupil::resource {
class Scene;
}

namespace Pupil::world {
class EmitterHelper {
public:
    EmitterHelper() noexcept;
    ~EmitterHelper() noexcept;

    void AddAreaEmitter(const resource::ShapeInstance &) noexcept;
    void AddEmitter(const resource::Emitter &) noexcept;

    void ComputeProbability() noexcept;

    void Clear() noexcept;
    optix::EmitterGroup GetEmitterGroup() noexcept;

private:
    bool m_dirty;

    std::vector<optix::Emitter> m_areas;
    std::vector<optix::Emitter> m_points;
    std::vector<optix::Emitter> m_directionals;
    optix::Emitter m_env;

    CUdeviceptr m_areas_cuda_memory;
    CUdeviceptr m_points_cuda_memory;
    CUdeviceptr m_directionals_cuda_memory;
    CUdeviceptr m_env_cuda_memory;
    CUdeviceptr m_env_cdf_weight_cuda_memory;
};
}// namespace Pupil::world