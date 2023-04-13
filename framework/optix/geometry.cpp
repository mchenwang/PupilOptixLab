#include "geometry.h"
#include "cuda/shape.h"

namespace Pupil::optix {
void Geometry::LoadGeometry(const scene::Shape &shape) noexcept {
    auto data_mngr = util::Singleton<cuda::CudaShapeDataManager>::instance();
    switch (shape.type) {
        case scene::EShapeType::_cube: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.cube.flip_normals;
            this->tri_mesh.flip_tex_coords = false;
            auto position_size = shape.cube.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(data_mngr->GetCudaMemPtr(shape.cube.positions, position_size), shape.cube.vertex_num * 3);
            auto normals_size = shape.cube.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(data_mngr->GetCudaMemPtr(shape.cube.normals, normals_size), shape.cube.vertex_num * 3);
            auto tex_size = shape.cube.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(data_mngr->GetCudaMemPtr(shape.cube.texcoords, tex_size), shape.cube.vertex_num * 3);
            auto idx_size = shape.cube.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(data_mngr->GetCudaMemPtr(shape.cube.indices, idx_size), shape.cube.face_num * 3);
        } break;
        case scene::EShapeType::_obj: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.obj.flip_normals;
            this->tri_mesh.flip_tex_coords = shape.obj.flip_tex_coords;
            auto position_size = shape.obj.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(data_mngr->GetCudaMemPtr(shape.obj.positions, position_size), shape.obj.vertex_num * 3);
            auto normals_size = shape.obj.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(data_mngr->GetCudaMemPtr(shape.obj.normals, normals_size), shape.obj.vertex_num * 3);
            auto tex_size = shape.obj.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(data_mngr->GetCudaMemPtr(shape.obj.texcoords, tex_size), shape.obj.vertex_num * 3);
            auto idx_size = shape.obj.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(data_mngr->GetCudaMemPtr(shape.obj.indices, idx_size), shape.obj.face_num * 3);
        } break;
        case scene::EShapeType::_rectangle: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.rect.flip_normals;
            this->tri_mesh.flip_tex_coords = false;
            auto position_size = shape.rect.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(data_mngr->GetCudaMemPtr(shape.rect.positions, position_size), shape.rect.vertex_num * 3);
            auto normals_size = shape.rect.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(data_mngr->GetCudaMemPtr(shape.rect.normals, normals_size), shape.rect.vertex_num * 3);
            auto tex_size = shape.rect.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(data_mngr->GetCudaMemPtr(shape.rect.texcoords, tex_size), shape.rect.vertex_num * 3);
            auto idx_size = shape.rect.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(data_mngr->GetCudaMemPtr(shape.rect.indices, idx_size), shape.rect.face_num * 3);
        } break;
        case scene::EShapeType::_sphere: {
            this->type = EType::Sphere;
            this->sphere.center = make_float3(shape.sphere.center.x, shape.sphere.center.y, shape.sphere.center.z);
            this->sphere.radius = shape.sphere.radius;
            this->sphere.flip_normal = shape.sphere.flip_normals;
        } break;
    }
}
}// namespace Pupil::optix