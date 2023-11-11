#include "scene/emitter.h"

#include "cuda/check.h"
#include "cuda/stream.h"

namespace Pupil {
    TriMeshEmitter::TriMeshEmitter(const util::CountableRef<resource::TriangleMesh>& shape,
                                   const Transform&                                  transform,
                                   const resource::TextureInstance&                  radiance) noexcept
        : Emitter(radiance, transform), m_data_dirty(true), m_cuda_memory(0), m_shape(shape) {
        m_num_face   = m_shape->GetFaceNum();
        m_mesh_index = std::make_unique<uint32_t[]>(m_num_face * 3);
        std::memcpy(m_mesh_index.get(), m_shape->GetIndex(), sizeof(uint32_t) * m_num_face * 3);

        m_num_vertex       = m_shape->GetVertexNum();
        m_mesh_vertex_ws   = std::make_unique<float[]>(m_num_vertex * 3);
        m_mesh_normal_ws   = std::make_unique<float[]>(m_num_vertex * 3);
        m_mesh_texcoord_ls = std::make_unique<float[]>(m_num_vertex * 2);
        m_areas            = std::make_unique<float[]>(m_num_face);

        SetTransform(m_transform);
    }

    TriMeshEmitter::~TriMeshEmitter() noexcept {
        m_shape.Reset();
        CUDA_FREE(m_cuda_memory);
    }

    struct TriMeshEmitterCudaMemory {
        CUdeviceptr indices;
        CUdeviceptr pos_ws;
        CUdeviceptr nor_ws;
        CUdeviceptr tex_ls;
        CUdeviceptr areas;
    };

    inline auto GetTriMeshEmitterCudaMemory(CUdeviceptr ptr, uint32_t num_face, uint32_t num_vertex) noexcept {
        TriMeshEmitterCudaMemory ret;
        ret.indices = ptr;
        ret.pos_ws  = ret.indices + sizeof(uint32_t) * num_face * 3;
        ret.nor_ws  = ret.pos_ws + sizeof(float) * num_vertex * 3;
        ret.tex_ls  = ret.nor_ws + sizeof(float) * num_vertex * 3;
        ret.areas   = ret.tex_ls + sizeof(float) * num_vertex * 2;
        return ret;
    }

    void TriMeshEmitter::UploadToCuda() noexcept {
        if (!m_data_dirty) return;

        auto size = sizeof(float) * 8 * m_num_vertex +  /*position + normal + texcoord*/
                    sizeof(uint32_t) * m_num_face * 3 + /*index*/
                    sizeof(float) * m_num_face;         /*area*/

        auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::EmitterUploading);

        if (!m_cuda_memory)
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_cuda_memory), size, *stream));

        auto [indices, pos_ws, nor_ws, tex_ls, areas] = GetTriMeshEmitterCudaMemory(m_cuda_memory, m_num_face, m_num_vertex);
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(indices), m_mesh_index.get(), sizeof(uint32_t) * m_num_face * 3, cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(pos_ws), m_mesh_vertex_ws.get(), sizeof(float) * m_num_vertex * 3, cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(nor_ws), m_mesh_normal_ws.get(), sizeof(float) * m_num_vertex * 3, cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(tex_ls), m_mesh_texcoord_ls.get(), sizeof(float) * m_num_vertex * 2, cudaMemcpyHostToDevice, *stream));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(areas), m_areas.get(), sizeof(float) * m_num_face, cudaMemcpyHostToDevice, *stream));

        m_radiance->UploadToCuda();

        m_data_dirty = false;
    }

    optix::Emitter TriMeshEmitter::GetOptixEmitter() noexcept {
        UploadToCuda();
        auto [indices, pos_ws, nor_ws, tex_ls, areas] = GetTriMeshEmitterCudaMemory(m_cuda_memory, m_num_face, m_num_vertex);
        optix::Emitter emitter;
        emitter.type = optix::EEmitterType::TriMesh;
        emitter.tri_mesh.indices.SetData(indices, m_num_face);
        emitter.tri_mesh.pos_ws.SetData(pos_ws, m_num_vertex);
        emitter.tri_mesh.nor_ws.SetData(nor_ws, m_num_vertex);
        emitter.tri_mesh.tex_ls.SetData(tex_ls, m_num_vertex);
        emitter.tri_mesh.areas.SetData(areas, m_num_face);
        emitter.tri_mesh.inv_num  = 1.f / m_num_face;
        emitter.tri_mesh.radiance = m_radiance.GetOptixTexture();
        return emitter;
    }

    void TriMeshEmitter::SetTransform(const Transform& trans) noexcept {
        m_transform           = trans;
        auto normal_transform = GetDiagonal3x3(Transpose(trans.Inverse().GetMatrix4x4()));
        auto vertex           = m_shape->GetVertex();
        auto normal           = m_shape->GetNormal();
        auto texcoord         = m_shape->GetTexcoord();
        for (auto i = 0u; i < m_num_face; ++i) {
            uint32_t idx[3];
            for (auto j = 0u; j < 3u; j++) idx[j] = m_mesh_index[i * 3 + j];

            Float3 p[3];
            for (auto j = 0u; j < 3u; j++) {
                p[j] = m_transform * Float3(vertex[idx[j] * 3], vertex[idx[j] * 3 + 1], vertex[idx[j] * 3 + 2]);

                m_mesh_vertex_ws[idx[j] * 3 + 0] = p[j].x;
                m_mesh_vertex_ws[idx[j] * 3 + 1] = p[j].y;
                m_mesh_vertex_ws[idx[j] * 3 + 2] = p[j].z;
            }

            Float3 n[3];
            if (normal) {
                for (auto j = 0u; j < 3u; j++) {
                    n[j] = Normalizef(normal_transform * Float3(normal[idx[j] * 3], normal[idx[j] * 3 + 1], normal[idx[j] * 3 + 2]));

                    m_mesh_normal_ws[idx[j] * 3 + 0] = n[j].x;
                    m_mesh_normal_ws[idx[j] * 3 + 1] = n[j].y;
                    m_mesh_normal_ws[idx[j] * 3 + 2] = n[j].z;
                }
            } else {
                auto v1 = p[0] - p[1];
                auto v2 = p[2] - p[1];
                n[0] = n[1] = n[2] = Normalizef(Cross(v2, v1));

                for (auto j = 0u; j < 3u; j++) {
                    m_mesh_normal_ws[idx[j] * 3 + 0] = n[j].x;
                    m_mesh_normal_ws[idx[j] * 3 + 1] = n[j].y;
                    m_mesh_normal_ws[idx[j] * 3 + 2] = n[j].z;
                }
            }

            for (auto j = 0u; j < 3u; j++) {
                m_mesh_texcoord_ls[idx[j] * 2 + 0] = texcoord ? texcoord[idx[j] * 2 + 0] : 0.f;
                m_mesh_texcoord_ls[idx[j] * 2 + 1] = texcoord ? texcoord[idx[j] * 2 + 1] : 0.f;
            }

            auto v1 = p[0] - p[1];
            auto v2 = p[2] - p[1];

            m_areas[i] = Lengthf(Cross(v1, v2)) * 0.5f;
        }

        m_data_dirty = true;
    }
}// namespace Pupil
