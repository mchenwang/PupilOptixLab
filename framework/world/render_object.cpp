#include "render_object.h"
#include "ias_manager.h"
#include "cuda/util.h"

#include "util/event.h"
#include "world.h"

namespace Pupil::world {

RenderObject::RenderObject(const resource::ShapeInstance &ins, unsigned int v_mask) noexcept
    : name(ins.name), transform(ins.transform), visibility_mask(v_mask) {
    Reset(ins.shape);

    mat.LoadMaterial(ins.mat);
}

void RenderObject::Reset(const resource::Shape *shape) noexcept {
    auto gas_mngr = util::Singleton<GASManager>::instance();
    auto [new_gas, is_reuse] = gas_mngr->RefGAS(shape);
    if (is_reuse) new_gas->Create();

    gas_mngr->Release(gas);
    gas = new_gas;

    shape_id = shape->id;
    aabb = shape->aabb;
    aabb.Transform(transform);
    is_emitter = is_emitter;

    if (shape->type == resource::EShapeType::_sphere) {
        geo.type = optix::Geometry::EType::Sphere;
        geo.sphere.center = make_float3(0.f);
        geo.sphere.radius = 1.f;
        geo.sphere.flip_normal = shape->sphere.flip_normals;
        sub_emitters_num = 1;
    } else if (shape->type == resource::EShapeType::_hair) {
        if ((shape->hair.flags & 0b11) == 0)
            geo.type = optix::Geometry::EType::LinearBSpline;
        else if ((shape->hair.flags & 0b11) == 1)
            geo.type = optix::Geometry::EType::QuadraticBSpline;
        else if ((shape->hair.flags & 0b11) == 2)
            geo.type = optix::Geometry::EType::CubicBSpline;
        else
            geo.type = optix::Geometry::EType::CatromSpline;

        auto device_memory =
            util::Singleton<resource::ShapeManager>::instance()->GetMeshDeviceMemory(shape);
        geo.curve.positions.SetData(device_memory.position, shape->hair.point_num * 3);
        geo.curve.indices.SetData(device_memory.index, shape->hair.segments_num);
    } else {
        geo.type = optix::Geometry::EType::TriMesh;
        auto device_memory =
            util::Singleton<resource::ShapeManager>::instance()->GetMeshDeviceMemory(shape);
        uint32_t vertex_num = shape->mesh.vertex_num;
        uint32_t face_num = shape->mesh.face_num;
        geo.tri_mesh.flip_normals = shape->mesh.flip_normals;
        geo.tri_mesh.flip_tex_coords = shape->mesh.flip_tex_coords;
        geo.tri_mesh.positions.SetData(device_memory.position, vertex_num * 3);
        geo.tri_mesh.normals.SetData(device_memory.normal, vertex_num * 3);
        geo.tri_mesh.texcoords.SetData(device_memory.texcoord, vertex_num * 2);
        geo.tri_mesh.indices.SetData(device_memory.index, face_num * 3);
        sub_emitters_num = face_num;
    }
    EventDispatcher<EWorldEvent::RenderInstanceUpdate>(this);
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