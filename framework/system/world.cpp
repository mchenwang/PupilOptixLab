#include "world.h"
#include "system.h"
#include "gui.h"

#include "util/event.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/camera.h"
#include "scene/scene.h"
#include "optix/scene/scene.h"

namespace Pupil {
void World::Init() noexcept {
    EventBinder<ECanvasEvent::MouseDragging>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;

        const struct {
            float x, y;
        } delta = *(decltype(delta) *)p;
        float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
        optix_scene->camera->Rotate(delta.x * scale, delta.y * scale);
        dirty = true;
    });

    EventBinder<ECanvasEvent::MouseWheel>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;

        float delta = *(float *)p;
        optix_scene->camera->SetFovDelta(delta);
        dirty = true;
    });

    EventBinder<ECanvasEvent::CameraMove>([this](void *p) {
        if (!util::Singleton<System>::instance()->render_flag) return;

        util::Float3 delta = *(util::Float3 *)p;
        optix_scene->camera->Move(delta * util::Camera::sensitivity * util::Camera::sensitivity_scale);
        dirty = true;
    });
}

void World::Destroy() noexcept {
    scene.reset();
    optix_scene.reset();
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
    timer.Stop();
    Pupil::Log::Info("Time consumed for scene loading: {:.3f}", timer.ElapsedSeconds());

    dirty = true;
    return true;
}

}// namespace Pupil