#include "render_object.h"
#include "ias_manager.h"
#include "cuda/util.h"

#include "util/event.h"
#include "world.h"

namespace Pupil::world {

RenderObject::RenderObject(const resource::ShapeInstance &ins, unsigned int v_mask) noexcept
    : name(ins.name), transform(ins.transform), visibility_mask(v_mask) {
    gas = util::Singleton<GASManager>::instance()->RefGAS(ins.shape);
    aabb = ins.shape->aabb;
    aabb.Transform(transform);
    is_emitter = ins.is_emitter;

    mat.LoadMaterial(ins.mat);

    if (ins.shape->type == resource::EShapeType::_sphere) {
        geo.type = optix::Geometry::EType::Sphere;
        geo.sphere.center = make_float3(0.f);
        geo.sphere.radius = 1.f;
        geo.sphere.flip_normal = ins.flip_normals;
        sub_emitters_num = 1;
    } else {
        geo.type = optix::Geometry::EType::TriMesh;
        auto device_memory =
            util::Singleton<resource::ShapeManager>::instance()->GetMeshDeviceMemory(ins.shape);
        uint32_t vertex_num = ins.shape->mesh.vertex_num;
        uint32_t face_num = ins.shape->mesh.face_num;
        geo.tri_mesh.flip_normals = ins.flip_normals;
        geo.tri_mesh.flip_tex_coords = ins.flip_tex_coords;
        geo.tri_mesh.positions.SetData(device_memory.position, vertex_num * 3);
        geo.tri_mesh.normals.SetData(device_memory.normal, vertex_num * 3);
        geo.tri_mesh.texcoords.SetData(device_memory.texcoord, vertex_num * 2);
        geo.tri_mesh.indices.SetData(device_memory.index, face_num * 3);
        sub_emitters_num = face_num;
    }
}

RenderObject::~RenderObject() noexcept {
    // util::Singleton<resource::ShapeManager>::instance()->Release(gas->ref_shape);
    util::Singleton<GASManager>::instance()->Release(gas);
}

void RenderObject::UpdateTransform(const util::Transform &new_transform) noexcept {
    transform = new_transform;
    EventDispatcher<EWorldEvent::RenderInstanceTransform>(this);
}

void RenderObject::ApplyTransform(const util::Transform &new_transform) noexcept {
    transform.matrix = new_transform.matrix * transform.matrix;
    EventDispatcher<EWorldEvent::RenderInstanceTransform>(this);
}
}// namespace Pupil::world