#include "world.h"
#include "gas_manager.h"

#include "system/system.h"
#include "system/gui/gui.h"

#include "util/event.h"
#include "util/log.h"
#include "util/timer.h"
#include "util/camera.h"
#include "resource/scene.h"

namespace Pupil::world {
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
        auto ro = reinterpret_cast<RenderObject *>(p);
        UpdateRenderObject(ro);
    });

    m_ias_manager = std::make_unique<IASManager>();
}

void World::Destroy() noexcept {
    m_ros.clear();
    scene.reset();
    camera.reset();
    emitters.reset();
    m_ias_manager.reset();
    util::Singleton<world::GASManager>::instance()->Destroy();
}

bool World::LoadScene(std::filesystem::path scene_file_path) noexcept {
    if (!std::filesystem::exists(scene_file_path)) {
        Pupil::Log::Warn("scene file [{}] does not exist.", scene_file_path.string());
        return false;
    }

    Pupil::Log::Info("start loading scene [{}].", scene_file_path.string());

    if (scene == nullptr) scene = std::make_unique<resource::Scene>();

    Pupil::Timer timer;
    timer.Start();
    scene->LoadFromXML(scene_file_path);
    LoadScene(scene.get());
    timer.Stop();
    Pupil::Log::Info("Time consumed for scene loading: {:.3f}s", timer.ElapsedSeconds());

    EventDispatcher<EWorldEvent::CameraChange>();
    return true;
}

bool World::LoadScene(resource::Scene *scene) noexcept {
    if (scene == nullptr) return false;
    m_ros.clear();
    m_ros.reserve(scene->shapes.size());

    for (auto &&shape : scene->shapes) {
        m_ros.emplace_back(std::make_unique<RenderObject>(shape, shape->transform, shape->id));
    }

    m_ias_manager->SetInstance(GetRenderobjects());

    auto &&sensor = scene->sensor;
    auto camera_desc = util::CameraDesc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };

    if (camera)
        camera->Reset(camera_desc);
    else
        camera = std::make_unique<CameraHelper>(camera_desc);

    if (emitters)
        emitters->Reset(scene);
    else
        emitters = std::make_unique<EmitterHelper>(scene);
    return true;
}

OptixTraversableHandle World::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
    return m_ias_manager->GetIASHandle(gas_offset, allow_update);
}

RenderObject *World::GetRenderObject(std::string_view id) const noexcept {
    for (auto &&ro : m_ros) {
        if (ro->id.compare(id) == 0)
            return ro.get();
    }

    Pupil::Log::Warn("Render Object [{}] missing.", id);
    return nullptr;
}

RenderObject *World::GetRenderObject(size_t index) const noexcept {
    if (index >= m_ros.size()) {
        Pupil::Log::Warn("#GetRenderObject index[{}] out of range[{}]", index, m_ros.size() - 1);
        return nullptr;
    }

    return m_ros[index].get();
}

std::vector<RenderObject *> World::GetRenderobjects() noexcept {
    std::vector<RenderObject *> render_objects;
    render_objects.reserve(m_ros.size());
    std::transform(m_ros.begin(), m_ros.end(), std::back_inserter(render_objects), [](const std::unique_ptr<RenderObject> &ro) { return ro.get(); });
    return render_objects;
}

void World::UpdateRenderObject(RenderObject *ro) noexcept {
    m_ias_manager->UpdateInstance(ro);
}

void World::SetDirty() noexcept {
    m_ias_manager->SetDirty();
}

bool World::IsDirty() const noexcept {
    return m_ias_manager->IsDirty();
}

void World::SetDirty(unsigned int gas_offset, bool allow_update) noexcept {
    m_ias_manager->SetDirty(gas_offset, allow_update);
}

bool World::IsDirty(unsigned int gas_offset, bool allow_update) const noexcept {
    return m_ias_manager->IsDirty(gas_offset, allow_update);
}

}// namespace Pupil::world