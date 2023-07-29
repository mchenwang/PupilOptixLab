#pragma once

#include "render/emitter.h"
#include "resource/emitter.h"
#include "resource/shape.h"

namespace Pupil::world {
class EmitterHelper {
public:
    EmitterHelper() noexcept;
    ~EmitterHelper() noexcept;

    size_t AddAreaEmitter(const resource::ShapeInstance &) noexcept;
    void ResetAreaEmitter(const resource::ShapeInstance &, size_t offset) noexcept;
    void AddEmitter(const resource::Emitter &) noexcept;

    void ComputeProbability() noexcept;

    void Clear() noexcept;
    optix::EmitterGroup GetEmitterGroup() noexcept;

private:
    void SetMeshAreaEmitter(const resource::ShapeInstance &, size_t offset) noexcept;
    void SetSphereAreaEmitter(const resource::ShapeInstance &, size_t offset) noexcept;

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