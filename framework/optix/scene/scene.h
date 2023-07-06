#pragma once

#include "camera.h"
#include "emitter.h"

#include <memory>

namespace Pupil::scene {
class Scene;
}

namespace Pupil::optix {
struct RenderObject;

class Scene {
    friend struct RenderObject;

public:
    util::CameraDesc camera_desc;

    std::unique_ptr<EmitterHelper> emitters;

    OptixTraversableHandle ias_handle = 0;

    Scene(Pupil::scene::Scene *, bool allow_update = true) noexcept;
    ~Scene() noexcept;

    void ResetScene(Pupil::scene::Scene *) noexcept;

    OptixTraversableHandle GetIASHandle() noexcept;

    RenderObject *GetRenderObject(std::string_view id) const noexcept;
    RenderObject *GetRenderObject(size_t index) const noexcept;

private:
    std::vector<std::unique_ptr<RenderObject>> m_ros;

    bool m_allow_update = false;
    bool m_scene_dirty = false;

    std::vector<OptixInstance> m_instances;
    CUdeviceptr m_instances_memory = 0;

    CUdeviceptr m_ias_buffer = 0;
    size_t m_ias_buffer_size = 0;

    CUdeviceptr m_ias_build_update_temp_buffer = 0;
    size_t m_ias_update_temp_buffer_size = 0;

    void CreateTopLevelAccel() noexcept;
};
}// namespace Pupil::optix