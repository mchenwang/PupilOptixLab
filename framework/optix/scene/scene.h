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
public:
    util::CameraDesc camera_desc;

    std::unique_ptr<EmitterHelper> emitters;

    OptixTraversableHandle ias_handle = 0;
    CUdeviceptr ias_buffer = 0;

    Scene(Pupil::scene::Scene *) noexcept;
    ~Scene() noexcept;

    void ResetScene(Pupil::scene::Scene *) noexcept;

private:
    std::vector<std::unique_ptr<RenderObject>> m_ros;

    void CreateTopLevelAccel() noexcept;
};
}// namespace Pupil::optix