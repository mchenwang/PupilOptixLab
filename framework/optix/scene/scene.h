#pragma once

#include "camera.h"
#include "emitter.h"

#include <memory>

namespace Pupil::resource {
class Scene;
}

namespace Pupil::optix {
struct RenderObject;
class IASManager;

class Scene {
    friend struct RenderObject;

public:
    util::CameraDesc camera_desc;

    std::unique_ptr<EmitterHelper> emitters;

    Scene(Pupil::resource::Scene *) noexcept;
    ~Scene() noexcept;

    void ResetScene(Pupil::resource::Scene *) noexcept;

    OptixTraversableHandle GetIASHandle(unsigned int gas_offset = 2, bool allow_update = false) noexcept;

    RenderObject *GetRenderObject(std::string_view id) const noexcept;
    RenderObject *GetRenderObject(size_t index) const noexcept;

    void UpdateRenderObject(RenderObject *) noexcept;

    std::vector<RenderObject *> GetRenderobjects() noexcept;

    void SetDirty() noexcept;
    bool IsDirty() const noexcept;
    void SetDirty(unsigned int gas_offset, bool allow_update) noexcept;
    bool IsDirty(unsigned int gas_offset, bool allow_update) const noexcept;

private:
    std::vector<std::unique_ptr<RenderObject>> m_ros;
    std::unique_ptr<IASManager> m_ias_manager;
};
}// namespace Pupil::optix