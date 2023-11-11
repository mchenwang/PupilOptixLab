#include "resource/shape.h"
#include "render/geometry.h"
#include "cuda/check.h"
#include "cuda/stream.h"

#include <optix_types.h>

namespace Pupil::resource {
    util::CountableRef<Shape> TriangleMesh::Make(std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<TriangleMesh>(UserDisableTag{}, name));
    }

    util::CountableRef<Shape> TriangleMesh::Make(std::string_view path, EShapeLoadFlag flags, std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->LoadShapeFromFile(path, flags, name);
    }

    TriangleMesh::TriangleMesh(UserDisableTag, std::string_view name) noexcept
        : Shape(name) {
        m_flip_normal   = false;
        m_flip_texcoord = true;

        m_num_vertex = 0;
        m_vertex     = nullptr;
        m_normal     = nullptr;
        m_texcoord   = nullptr;
        m_num_face   = 0;
        m_index      = nullptr;

        m_device_memory_index     = 0;
        m_device_memory_normal    = 0;
        m_device_memory_sbt_index = 0;
        m_device_memory_texcoord  = 0;
        m_device_memory_vertex    = 0;
    }

    TriangleMesh::~TriangleMesh() noexcept {
        m_vertex.reset();
        m_normal.reset();
        m_texcoord.reset();
        m_index.reset();
    }

    void* TriangleMesh::Clone() const noexcept {
        auto clone = new TriangleMesh(UserDisableTag{}, m_name);
        clone->SetVertex(m_vertex.get(), m_num_vertex);
        clone->SetIndex(m_index.get(), m_num_face);
        clone->SetNormal(m_normal.get(), m_num_vertex);
        clone->SetTexcoord(m_texcoord.get(), m_num_vertex);
        clone->SetFlipNormal(m_flip_normal);
        clone->SetFlipTexcoord(m_flip_texcoord);

        return clone;
    }

    uint64_t TriangleMesh::GetMemorySizeInByte() const noexcept {
        auto vertex   = sizeof(float) * 3 * m_num_vertex;
        auto index    = sizeof(uint32_t) * 3 * m_num_face;
        auto normal   = sizeof(float) * 3 * m_num_vertex;
        auto texcoord = sizeof(float) * 2 * m_num_vertex;
        return vertex + index + normal + texcoord;
    }

    void TriangleMesh::UploadToCuda() noexcept {
        assert(m_index && m_vertex);
        if (m_data_dirty) {
            auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::ShapeUploading);

            CUDA_FREE_ASYNC(m_device_memory_vertex, *stream);
            CUDA_FREE_ASYNC(m_device_memory_index, *stream);
            CUDA_FREE_ASYNC(m_device_memory_normal, *stream);
            CUDA_FREE_ASYNC(m_device_memory_texcoord, *stream);
            CUDA_FREE_ASYNC(m_device_memory_sbt_index, *stream);

            auto size_vertex = sizeof(float) * 3 * m_num_vertex;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_vertex), size_vertex, *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_vertex), m_vertex.get(), size_vertex, cudaMemcpyHostToDevice, *stream));

            auto size_index = sizeof(uint32_t) * m_num_face * 3;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_index), size_index, *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_index), m_index.get(), size_index, cudaMemcpyHostToDevice, *stream));

            if (m_normal) {
                auto size_normal = sizeof(float) * 3 * m_num_vertex;
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_normal), size_normal, *stream));
                CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_normal), m_normal.get(), size_normal, cudaMemcpyHostToDevice, *stream));
            }

            if (m_texcoord) {
                auto size_texcoord = sizeof(float) * 2 * m_num_vertex;
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_texcoord), size_texcoord, *stream));
                CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_texcoord), m_texcoord.get(), size_texcoord, cudaMemcpyHostToDevice, *stream));
            }

            uint32_t sbt_index = 0;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_sbt_index), sizeof(sbt_index), *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_sbt_index), &sbt_index, sizeof(sbt_index), cudaMemcpyHostToDevice, *stream));

            m_upload_event->Reset(stream.Get());

            m_data_dirty = false;
        }
    }

    optix::Geometry TriangleMesh::GetOptixGeometry() noexcept {
        optix::Geometry geo;
        geo.type                     = optix::Geometry::EType::TriMesh;
        geo.tri_mesh.flip_normals    = m_flip_normal;
        geo.tri_mesh.flip_tex_coords = m_flip_texcoord;

        geo.tri_mesh.positions.SetData(m_device_memory_vertex, m_num_vertex);
        geo.tri_mesh.indices.SetData(m_device_memory_index, m_num_face);
        if (m_normal)
            geo.tri_mesh.normals.SetData(m_device_memory_normal, m_num_vertex);
        else
            geo.tri_mesh.normals.SetData(0, 0);

        if (m_texcoord)
            geo.tri_mesh.texcoords.SetData(m_device_memory_texcoord, m_num_vertex);
        else
            geo.tri_mesh.texcoords.SetData(0, 0);

        return geo;
    }

    OptixBuildInput TriangleMesh::GetOptixBuildInput() noexcept {
        OptixBuildInput input{};
        input.type          = OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray = {
            .vertexBuffers       = &m_device_memory_vertex,
            .numVertices         = static_cast<unsigned int>(m_num_vertex),
            .vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3,
            .vertexStrideInBytes = sizeof(float) * 3,
            .indexBuffer         = m_device_memory_index,
            .numIndexTriplets    = static_cast<unsigned int>(m_num_face),
            .indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
            .indexStrideInBytes  = sizeof(unsigned int) * 3,
            // .preTransform = d_transform,
            .flags                       = &Shape::s_input_flag,
            .numSbtRecords               = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
            .sbtIndexOffsetBuffer        = m_device_memory_sbt_index,
            .sbtIndexOffsetSizeInBytes   = sizeof(uint32_t),
            .sbtIndexOffsetStrideInBytes = sizeof(uint32_t),
            // .transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12
        };

        return input;
    }

    void TriangleMesh::SetVertex(const float* vertex, uint32_t num_vertex) noexcept {
        assert(vertex != nullptr && num_vertex > 0);

        if (num_vertex != m_num_vertex) {
            m_vertex.reset();
            m_normal.reset();
            m_texcoord.reset();
            m_vertex     = std::make_unique<float[]>(num_vertex * 3);
            m_num_vertex = num_vertex;
        }
        aabb = util::AABB{};
        for (int i = 0; i < num_vertex; ++i)
            aabb.Merge(Float3(vertex[i * 3 + 0], vertex[i * 3 + 1], vertex[i * 3 + 2]));

        std::memcpy(m_vertex.get(), vertex, sizeof(float) * 3 * num_vertex);
        m_data_dirty = true;
    }

    void TriangleMesh::SetIndex(const uint32_t* index, uint32_t num_face) noexcept {
        assert(index != nullptr && num_face > 0);

        if (num_face != m_num_face) {
            m_index.reset();
            m_index    = std::make_unique<uint32_t[]>(num_face * 3);
            m_num_face = num_face;
        }

        std::memcpy(m_index.get(), index, sizeof(uint32_t) * 3 * num_face);
        m_data_dirty = true;
    }

    void TriangleMesh::SetNormal(const float* normal, uint32_t num_vertex) noexcept {
        if (normal == nullptr) {
            m_normal.reset();
        } else {
            assert(m_num_vertex == num_vertex);
            if (m_normal == nullptr)
                m_normal = std::make_unique<float[]>(num_vertex * 3);

            std::memcpy(m_normal.get(), normal, sizeof(float) * 3 * num_vertex);
        }

        m_data_dirty = true;
    }

    void TriangleMesh::SetTexcoord(const float* texcoord, uint32_t num_vertex) noexcept {
        if (texcoord == nullptr) {
            m_texcoord.reset();
        } else {
            assert(m_num_vertex == num_vertex);
            if (m_texcoord == nullptr)
                m_texcoord = std::make_unique<float[]>(num_vertex * 2);

            std::memcpy(m_texcoord.get(), texcoord, sizeof(float) * 2 * num_vertex);
        }

        m_data_dirty = true;
    }

    void TriangleMesh::SetFlipNormal(bool flip_normal) noexcept {
        m_flip_normal = flip_normal;
    }

    void TriangleMesh::SetFlipTexcoord(bool flip_texcoord) noexcept {
        m_flip_texcoord = flip_texcoord;
    }

}// namespace Pupil::resource