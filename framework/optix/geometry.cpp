#include "geometry.h"
#include "cuda/util.h"

#include <unordered_map>

namespace {
std::unordered_map<void *, CUdeviceptr> m_cuda_geometry_map;

inline CUdeviceptr GetCudaMemory(void *data, size_t size) noexcept {
    if (data == nullptr || size == 0) return 0;
    auto it = m_cuda_geometry_map.find(data);
    if (it == m_cuda_geometry_map.end()) {
        m_cuda_geometry_map[data] = Pupil::cuda::CudaMemcpyToDevice(data, size);
    }
    return m_cuda_geometry_map[data];
}
}// namespace

namespace Pupil::optix {
void Geometry::LoadGeometry(const scene::Shape &shape) noexcept {
    switch (shape.type) {
        case scene::EShapeType::_cube: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.cube.flip_normals;
            this->tri_mesh.flip_tex_coords = false;
            auto position_size = shape.cube.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(GetCudaMemory(shape.cube.positions, position_size), shape.cube.vertex_num * 3);
            auto normals_size = shape.cube.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(GetCudaMemory(shape.cube.normals, normals_size), shape.cube.vertex_num * 3);
            auto tex_size = shape.cube.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(GetCudaMemory(shape.cube.texcoords, tex_size), shape.cube.vertex_num * 3);
            auto idx_size = shape.cube.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(GetCudaMemory(shape.cube.indices, idx_size), shape.cube.face_num * 3);
        } break;
        case scene::EShapeType::_obj: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.obj.flip_normals;
            this->tri_mesh.flip_tex_coords = shape.obj.flip_tex_coords;
            auto position_size = shape.obj.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(GetCudaMemory(shape.obj.positions, position_size), shape.obj.vertex_num * 3);
            auto normals_size = shape.obj.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(GetCudaMemory(shape.obj.normals, normals_size), shape.obj.vertex_num * 3);
            auto tex_size = shape.obj.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(GetCudaMemory(shape.obj.texcoords, tex_size), shape.obj.vertex_num * 3);
            auto idx_size = shape.obj.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(GetCudaMemory(shape.obj.indices, idx_size), shape.obj.face_num * 3);
        } break;
        case scene::EShapeType::_rectangle: {
            this->type = EType::TriMesh;
            this->tri_mesh.flip_normals = shape.rect.flip_normals;
            this->tri_mesh.flip_tex_coords = false;
            auto position_size = shape.rect.vertex_num * 3 * sizeof(float);
            this->tri_mesh.positions.SetData(GetCudaMemory(shape.rect.positions, position_size), shape.rect.vertex_num * 3);
            auto normals_size = shape.rect.vertex_num * 3 * sizeof(float);
            this->tri_mesh.normals.SetData(GetCudaMemory(shape.rect.normals, normals_size), shape.rect.vertex_num * 3);
            auto tex_size = shape.rect.vertex_num * 2 * sizeof(float);
            this->tri_mesh.texcoords.SetData(GetCudaMemory(shape.rect.texcoords, tex_size), shape.rect.vertex_num * 3);
            auto idx_size = shape.rect.face_num * 3 * sizeof(uint32_t);
            this->tri_mesh.indices.SetData(GetCudaMemory(shape.rect.indices, idx_size), shape.rect.face_num * 3);
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