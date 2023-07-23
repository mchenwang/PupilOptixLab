#include "world.h"
#include "system.h"
#include "gui/gui.h"

#include "util/event.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/camera.h"
#include "scene/scene.h"
#include "optix/scene/scene.h"
#include "optix/scene/mesh.h"

namespace Pupil {
void World::Init() noexcept {
    EventBinder<ECanvasEvent::MouseDragging>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag || !camera) return;

        const struct {
            float x, y;
        } delta = *(decltype(delta) *)p;
        float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
        camera->Rotate(delta.x * scale, delta.y * scale);
        EventDispatcher<EWorldEvent::CameraViewChange>();
        EventDispatcher<EWorldEvent::CameraChange>();
    });

    EventBinder<ECanvasEvent::MouseWheel>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag || !camera) return;

        float delta = *(float *)p;
        camera->SetFovDelta(delta);
        EventDispatcher<EWorldEvent::CameraFovChange>();
        EventDispatcher<EWorldEvent::CameraChange>();
    });

    EventBinder<ECanvasEvent::CameraMove>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag || !camera) return;

        util::Float3 delta = *(util::Float3 *)p;
        camera->Move(delta * util::Camera::sensitivity * util::Camera::sensitivity_scale);
        EventDispatcher<EWorldEvent::CameraMove>();
        EventDispatcher<EWorldEvent::CameraChange>();
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *p) {
        auto ro = reinterpret_cast<optix::RenderObject *>(p);
        if (optix_scene) {
            optix_scene->UpdateRenderObject(ro);
        }
    });
}

void World::Destroy() noexcept {
    scene.reset();
    optix_scene.reset();
    camera.reset();
    util::Singleton<optix::MeshManager>::instance()->Destroy();
}

bool World::LoadScene(std::filesystem::path scene_file_path) noexcept {
    if (!std::filesystem::exists(scene_file_path)) {
        Pupil::Log::Warn("scene file [{}] does not exist.", scene_file_path.string());
        return false;
    }

    Pupil::Log::Info("start loading scene [{}].", scene_file_path.string());

    if (scene == nullptr) scene = std::make_unique<scene::Scene>();

    Pupil::Timer timer;
    timer.Start();
    scene->LoadFromXML(scene_file_path);
    if (optix_scene)
        optix_scene->ResetScene(scene.get());
    else
        optix_scene = std::make_unique<optix::Scene>(scene.get());

    if (camera)
        camera->Reset(optix_scene->camera_desc);
    else
        camera = std::make_unique<CameraHelper>(optix_scene->camera_desc);
    timer.Stop();
    Pupil::Log::Info("Time consumed for scene loading: {:.3f}s", timer.ElapsedSeconds());

    EventDispatcher<EWorldEvent::CameraChange>();
    return true;
}

}// namespace Pupil