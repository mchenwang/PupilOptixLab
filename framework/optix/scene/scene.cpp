#include "ias_manager.h"
#include "mesh.h"
#include "render_object.h"
#include "optix/scene/scene.h"
#include "scene/scene.h"

#include "optix/context.h"
#include "optix/check.h"
#include "cuda/util.h"

#include "system/world.h"
#include "util/event.h"

#include <optix_stubs.h>

namespace Pupil::optix {
Scene::Scene(Pupil::scene::Scene *scene) noexcept {
    m_ias_manager = std::make_unique<IASManager>();
    ResetScene(scene);
}

void Scene::ResetScene(Pupil::scene::Scene *scene) noexcept {
    m_ros.clear();
    m_ros.reserve(scene->shapes.size());

    MeshEntity temp_mesh{};
    SphereEntity temp_sphere{};

    for (auto &&shape : scene->shapes) {
        switch (shape.type) {
            case scene::EShapeType::_obj: {
                temp_mesh.vertex_num = shape.obj.vertex_num;
                temp_mesh.vertices = shape.obj.positions;
                temp_mesh.index_triplets_num = shape.obj.face_num;
                temp_mesh.indices = shape.obj.indices;
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.transform, shape.id));
            } break;
            case scene::EShapeType::_rectangle: {
                temp_mesh.vertex_num = shape.rect.vertex_num;
                temp_mesh.vertices = shape.rect.positions;
                temp_mesh.index_triplets_num = shape.rect.face_num;
                temp_mesh.indices = shape.rect.indices;
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.transform, shape.id));
            } break;
            case scene::EShapeType::_cube: {
                temp_mesh.vertex_num = shape.cube.vertex_num;
                temp_mesh.vertices = shape.cube.positions;
                temp_mesh.index_triplets_num = shape.cube.face_num;
                temp_mesh.indices = shape.cube.indices;
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::Custom, &temp_mesh, shape.transform, shape.id));
            } break;
            case scene::EShapeType::_sphere: {
                // temp_sphere.center = make_float3(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                // temp_sphere.radius = shape.sphere.radius;
                util::Transform sphere_init_trans;
                sphere_init_trans.Scale(shape.sphere.radius, shape.sphere.radius, shape.sphere.radius);
                sphere_init_trans.Translate(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
                temp_sphere.center = make_float3(0.f);
                temp_sphere.radius = 1.f;
                sphere_init_trans.matrix = shape.transform.matrix * sphere_init_trans.matrix;
                m_ros.emplace_back(std::make_unique<RenderObject>(EMeshEntityType::BuiltinSphere, &temp_sphere, sphere_init_trans, shape.id));
            } break;
        }
    }

    m_ias_manager->SetInstance(GetRenderobjects());

    auto &&sensor = scene->sensor;
    camera_desc = util::CameraDesc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };

    if (!emitters)
        emitters = std::make_unique<optix::EmitterHelper>(scene);
    else
        emitters->Reset(scene);
}

OptixTraversableHandle Scene::GetIASHandle(unsigned int gas_offset, bool allow_update) noexcept {
    return m_ias_manager->GetIASHandle(gas_offset, allow_update);
}

RenderObject *Scene::GetRenderObject(std::string_view id) const noexcept {
    for (auto &&ro : m_ros) {
        if (ro->id.compare(id) == 0)
            return ro.get();
    }

    Pupil::Log::Warn("Render Object [{}] missing.", id);
    return nullptr;
}
RenderObject *Scene::GetRenderObject(size_t index) const noexcept {
    if (index >= m_ros.size()) {
        Pupil::Log::Warn("#GetRenderObject index[{}] out of range[{}]", index, m_ros.size() - 1);
        return nullptr;
    }

    return m_ros[index].get();
}

std::vector<RenderObject *> Scene::GetRenderobjects() noexcept {
    std::vector<RenderObject *> render_objects;
    render_objects.reserve(m_ros.size());
    std::transform(m_ros.begin(), m_ros.end(), std::back_inserter(render_objects), [](const std::unique_ptr<RenderObject> &ro) { return ro.get(); });
    return render_objects;
}

void Scene::UpdateRenderObject(RenderObject *ro) noexcept {
    m_ias_manager->UpdateInstance(ro);
}

void Scene::SetDirty() noexcept {
    m_ias_manager->SetDirty();
}

bool Scene::IsDirty() const noexcept {
    return m_ias_manager->IsDirty();
}

void Scene::SetDirty(unsigned int gas_offset, bool allow_update) noexcept {
    m_ias_manager->SetDirty(gas_offset, allow_update);
}

bool Scene::IsDirty(unsigned int gas_offset, bool allow_update) const noexcept {
    return m_ias_manager->IsDirty(gas_offset, allow_update);
}

Scene::~Scene() noexcept {
    m_ias_manager.reset();
    emitters.reset();
}
}// namespace Pupil::optix