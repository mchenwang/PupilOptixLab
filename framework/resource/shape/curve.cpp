#include "resource/shape.h"
#include "render/geometry.h"
#include "cuda/check.h"
#include "cuda/stream.h"

#include <optix_types.h>

#include <algorithm>

namespace Pupil::resource {
    util::CountableRef<Shape> Curve::Make(EType type, std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<Curve>(UserDisableTag{}, type, name));
    }

    Curve::Curve(UserDisableTag, EType type, std::string_view name) noexcept
        : Shape(name), m_curve_type(type) {
        m_num_ctrl_vertex                         = 0;
        m_num_segment_vertex_index                = 0;
        m_num_strand                              = 0;
        m_max_width                               = 0.f;
        m_ctrl_vertex                             = nullptr;
        m_width                                   = nullptr;
        m_segment_ctrl_vertex_index               = nullptr;
        m_strand_head_ctrl_vertex_index           = nullptr;
        m_device_memory_ctrl_vertex               = 0;
        m_device_memory_width                     = 0;
        m_device_memory_segment_ctrl_vertex_index = 0;
    }

    Curve::~Curve() noexcept {
        m_ctrl_vertex.reset();
        m_width.reset();
        m_segment_ctrl_vertex_index.reset();
        m_strand_head_ctrl_vertex_index.reset();
    }

    void* Curve::Clone() const noexcept {
        auto clone = new Curve(UserDisableTag{}, m_curve_type, m_name);
        clone->SetCtrlVertex(m_ctrl_vertex.get(), m_num_ctrl_vertex);
        clone->SetWidth(m_width.get(), m_num_ctrl_vertex);
        clone->SetStrandHeadCtrlVertexIndex(m_strand_head_ctrl_vertex_index.get(), m_num_strand);
        return clone;
    }

    uint64_t Curve::GetMemorySizeInByte() const noexcept {
        auto ctrl_vertex = sizeof(float) * 3 * m_num_ctrl_vertex;
        auto width       = sizeof(float) * m_num_ctrl_vertex;
        auto strand      = sizeof(uint32_t) * m_num_strand;
        auto segment     = sizeof(uint32_t) * m_num_segment_vertex_index;
        return ctrl_vertex + width + strand + segment;
    }

    void Curve::UploadToCuda() noexcept {
        if (m_data_dirty) {
            auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::ShapeUploading);
            CUDA_FREE_ASYNC(m_device_memory_ctrl_vertex, *stream);
            CUDA_FREE_ASYNC(m_device_memory_width, *stream);
            CUDA_FREE_ASYNC(m_device_memory_segment_ctrl_vertex_index, *stream);

            auto size_vertex = sizeof(float) * 3 * m_num_ctrl_vertex;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_ctrl_vertex), size_vertex, *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_ctrl_vertex), m_ctrl_vertex.get(), size_vertex, cudaMemcpyHostToDevice, *stream));

            auto size_width = sizeof(float) * m_num_ctrl_vertex;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_width), size_width, *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_device_memory_width), m_width.get(), size_width, cudaMemcpyHostToDevice, *stream));

            auto size_index = sizeof(uint32_t) * m_num_segment_vertex_index;
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_segment_ctrl_vertex_index), size_index, *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void**>(m_device_memory_segment_ctrl_vertex_index), m_segment_ctrl_vertex_index.get(), size_index, cudaMemcpyHostToDevice, *stream));

            m_upload_event->Reset(stream.Get());
            m_data_dirty = false;
        }
    }

    util::AABB Curve::GetAABB() const noexcept {
        util::AABB ret = aabb;
        ret.min -= util::Float3(m_max_width);
        ret.max += util::Float3(m_max_width);
        return ret;
    }

    optix::Geometry Curve::GetOptixGeometry() noexcept {
        optix::Geometry geo;
        switch (m_curve_type) {
            case EType::Linear:
                geo.type = optix::Geometry::EType::LinearBSpline;
                break;
            case EType::Quadratic:
                geo.type = optix::Geometry::EType::QuadraticBSpline;
                break;
            case EType::Cubic:
                geo.type = optix::Geometry::EType::CubicBSpline;
                break;
            case EType::Catrom:
                geo.type = optix::Geometry::EType::CatromSpline;
                break;
        }
        return geo;
    }

    OptixBuildInput Curve::GetOptixBuildInput() noexcept {
        OptixBuildInput input{};
        input.type       = OPTIX_BUILD_INPUT_TYPE_CURVES;
        input.curveArray = {
            .numPrimitives        = m_num_segment_vertex_index,
            .vertexBuffers        = &m_device_memory_ctrl_vertex,
            .numVertices          = m_num_ctrl_vertex,
            .vertexStrideInBytes  = sizeof(float3),
            .widthBuffers         = &m_device_memory_width,
            .widthStrideInBytes   = sizeof(float),
            .normalBuffers        = 0,
            .normalStrideInBytes  = 0,
            .indexBuffer          = m_device_memory_segment_ctrl_vertex_index,
            .indexStrideInBytes   = sizeof(uint32_t),
            .flag                 = OPTIX_GEOMETRY_FLAG_NONE,
            .primitiveIndexOffset = 0};

        switch (m_curve_type) {
            case EType::Linear:
                input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
                break;
            case EType::Quadratic:
                input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
                break;
            case EType::Cubic:
                input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
                break;
            case EType::Catrom:
                input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
                break;
        }

        return input;
    }

    void Curve::SetCurveType(EType type) noexcept {
        m_curve_type = type;
    }

    void Curve::SetCtrlVertex(const float* ctrl_vertex, uint32_t num_ctrl_vertex) noexcept {
        assert(ctrl_vertex != nullptr && num_ctrl_vertex > 0);

        if (num_ctrl_vertex != m_num_ctrl_vertex) {
            m_ctrl_vertex.reset();
            m_width.reset();

            m_ctrl_vertex     = std::make_unique<float[]>(num_ctrl_vertex * 3);
            m_width           = std::make_unique<float[]>(num_ctrl_vertex);
            m_num_ctrl_vertex = num_ctrl_vertex;
        }

        aabb = util::AABB{};
        for (int i = 0; i < m_num_ctrl_vertex; i += 3) {
            aabb.Merge(util::Float3(ctrl_vertex[i], ctrl_vertex[i + 1], ctrl_vertex[i + 2]));
        }

        std::memcpy(m_ctrl_vertex.get(), ctrl_vertex, sizeof(float) * 3 * num_ctrl_vertex);
        m_data_dirty = true;
    }

    void Curve::SetWidth(float width) noexcept {
        SetWidth(&width, 1);
    }

    void Curve::SetWidth(const float* width, uint32_t num_width) noexcept {
        assert(width != nullptr && num_width > 0);

        if (num_width == 1) {
            std::fill_n(m_width.get(), m_num_ctrl_vertex, *width);
            m_max_width = *width;
        } else {
            assert(num_width == m_num_ctrl_vertex);

            m_max_width = *std::max_element(width, width + num_width);
            std::memcpy(m_width.get(), width, sizeof(float) * m_num_ctrl_vertex);
        }

        m_data_dirty = true;
    }

    void Curve::SetStrandHeadCtrlVertexIndex(const uint32_t* strand_vertex_index, uint32_t num_strand) noexcept {
        assert(strand_vertex_index != nullptr && num_strand > 0);

        if (num_strand != m_num_strand) {
            m_strand_head_ctrl_vertex_index.reset();
            m_segment_ctrl_vertex_index.reset();

            m_strand_head_ctrl_vertex_index = std::make_unique<uint32_t[]>(num_strand);
            m_num_strand                    = num_strand;
        }
        std::memcpy(m_strand_head_ctrl_vertex_index.get(), strand_vertex_index, sizeof(uint32_t) * num_strand);

        auto                  curve_degree = GetDegree();
        std::vector<uint32_t> segments;
        for (int i = 0; i < num_strand - 1; ++i) {
            const uint32_t start = strand_vertex_index[i];
            const uint32_t end   = strand_vertex_index[i + 1] - curve_degree;
            for (uint32_t segment_index = start; segment_index < end; ++segment_index)
                segments.push_back(segment_index);
        }
        m_num_segment_vertex_index  = static_cast<uint32_t>(segments.size());
        m_segment_ctrl_vertex_index = std::make_unique<uint32_t[]>(segments.size());
        std::memcpy(m_segment_ctrl_vertex_index.get(), segments.data(), sizeof(uint32_t) * segments.size());

        m_data_dirty = true;
    }
}// namespace Pupil::resource