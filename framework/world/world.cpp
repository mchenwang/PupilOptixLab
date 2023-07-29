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

    EventBinder<EWorldEvent::RenderInstanceTransform>([this](void *p) {
        auto ro = reinterpret_cast<RenderObject *>(p);
        auto &ins = scene->shape_instances[m_ro_in_scene_index[ro]];
        ins.transform = ro->transform;
        if (ins.is_emitter) {
            emitters->ResetAreaEmitter(ins, m_ro_emitter_offset[ro]);
            emitters->ComputeProbability();
        }
        EventDispatcher<EWorldEvent::RenderInstanceUpdate>(p);
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *p) {
        auto ro = reinterpret_cast<RenderObject *>(p);
        UpdateRenderObject(ro);
    });

    m_ias_manager = std::make_unique<IASManager>();
    camera = std::make_unique<CameraHelper>();
    emitters = std::make_unique<EmitterHelper>();
    scene = std::make_unique<resource::Scene>();
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

    Pupil::Timer timer;
    timer.Start();
    m_ros.clear();
    if (!scene->LoadFromXML(scene_file_path) || !LoadScene(scene.get())) {
        Pupil::Log::Error("Scene load failed: {}.", scene_file_path.string().c_str());
        return false;
    }
    timer.Stop();
    Pupil::Log::Info("Time consumed for scene loading: {:.3f}s", timer.ElapsedSeconds());

    util::Singleton<GASManager>::instance()->ClearDanglingMemory();
    util::Singleton<resource::ShapeManager>::instance()->ClearDanglingMemory();

    EventDispatcher<EWorldEvent::CameraChange>();
    return true;
}

bool World::LoadScene(resource::Scene *scene) noexcept {
    if (scene == nullptr) return false;

    auto &&sensor = scene->sensor;
    auto camera_desc = util::CameraDesc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };

    camera->Reset(camera_desc);

    m_ros.clear();
    m_ros.reserve(scene->shape_instances.size());

    emitters->Clear();
    size_t emitter_offset = 0;
    for (size_t index = 0; index < scene->shape_instances.size(); ++index) {
        auto &ins = scene->shape_instances[index];
        if (!ins.shape || ins.shape->type == resource::EShapeType::_unknown) continue;
        m_ros.emplace_back(std::make_unique<RenderObject>(ins));
        m_ro_in_scene_index[m_ros.back().get()] = index;

        if (ins.is_emitter) {
            m_ro_emitter_offset[m_ros.back().get()] = emitter_offset;
            emitter_offset = emitters->AddAreaEmitter(ins);
        }
    }

    for (auto &&emitter : scene->emitters) {
        emitters->AddEmitter(emitter);
    }
    emitters->ComputeProbability();

    m_ias_manager->SetInstance(GetRenderobjects());
    return true;
}

OptixTraversableHandle World::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
    return m_ias_manager->GetIASHandle(gas_offset, allow_update);
}

RenderObject *World::GetRenderObject(std::string_view name) const noexcept {
    for (auto &&ro : m_ros) {
        if (ro->name.compare(name) == 0)
            return ro.get();
    }

    Pupil::Log::Warn("Render Object [{}] missing.", name);
    return nullptr;
}

RenderObject *World::GetRenderObject(size_t index) const noexcept {
    if (index >= m_ros.size()) {
        Pupil::Log::Warn("#GetRenderObject index[{}] out of range[{}]", index, m_ros.size() - 1);
        return nullptr;
    }

    return m_ros[index].get();
}

void World::RemoveRenderObject(std::string_view name) noexcept {
    for (auto it = m_ros.begin(); it != m_ros.end(); ++it) {
        if ((*it)->name.compare(name) == 0) {
            EventDispatcher<EWorldEvent::RenderInstanceRemove>((*it).get());
            m_ros.erase(it);
            m_ias_manager->SetInstance(GetRenderobjects());
            return;
        }
    }
}

void World::RemoveRenderObject(size_t index) noexcept {
    if (index >= m_ros.size()) return;
    EventDispatcher<EWorldEvent::RenderInstanceRemove>(m_ros[index].get());
    m_ros.erase(m_ros.begin() + index);
    m_ias_manager->SetInstance(GetRenderobjects());
}

std::vector<RenderObject *> World::GetRenderobjects() noexcept {
    std::vector<RenderObject *> render_objects;
    render_objects.reserve(m_ros.size());
    std::transform(m_ros.begin(), m_ros.end(), std::back_inserter(render_objects), [](const std::unique_ptr<RenderObject> &ro) { return ro.get(); });
    return render_objects;
}

util::AABB World::GetAABB() noexcept {
    util::AABB aabb;
    for (auto &&ro : m_ros)
        aabb.Merge(ro->aabb);

    return aabb;
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