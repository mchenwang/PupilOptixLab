#include "../shape.h"
#include "render/geometry.h"
#include "cuda/check.h"
#include "cuda/stream.h"

#include <optix_types.h>

namespace Pupil::resource {
    util::CountableRef<Shape> Sphere::Make(std::string_view name) noexcept {
        auto shape_mngr = util::Singleton<ShapeManager>::instance();
        return shape_mngr->Register(std::make_unique<Sphere>(UserDisableTag{}, name));
    }

    inline util::AABB SetAABB(const util::Float3& c, float r) noexcept {
        util::AABB aabb;
        aabb.min = c - r;
        aabb.max = c + r;
        return aabb;
    }

    Sphere::Sphere(UserDisableTag, std::string_view name) noexcept
        : Shape(name), m_device_memory_radius(0), m_device_memory_center(0), m_device_memory_sbt_index(0) {
        m_center      = util::Float3(0.f);
        m_radius      = 1.f;
        m_flip_normal = false;
        aabb          = SetAABB(m_center, m_radius);
    }

    Sphere::Sphere(UserDisableTag, const util::Float3& c, float r, std::string_view name) noexcept
        : Shape(name), m_center(c), m_radius(r),
          m_device_memory_radius(0), m_device_memory_center(0), m_device_memory_sbt_index(0) {
        m_flip_normal = false;
        aabb          = SetAABB(m_center, m_radius);
    }

    Sphere::~Sphere() noexcept {
    }

    void* Sphere::Clone() const noexcept {
        auto clone           = new Sphere(UserDisableTag{}, m_center, m_radius, m_name);
        clone->m_flip_normal = m_flip_normal;
        return clone;
    }

    uint64_t Sphere::GetMemorySizeInByte() const noexcept { return sizeof(m_center) + sizeof(m_radius); }

    void Sphere::SetCenter(const util::Float3& center) noexcept {
        m_center = center;
        aabb     = SetAABB(m_center, m_radius);

        m_data_dirty = true;
    }

    void Sphere::SetRadius(float radius) noexcept {
        m_radius = radius;
        aabb     = SetAABB(m_center, m_radius);

        m_data_dirty = true;
    }

    void Sphere::SetFlipNormal(bool flip_normal) noexcept {
        m_flip_normal = flip_normal;
    }

    void Sphere::UploadToCuda() noexcept {
        if (m_data_dirty) {
            auto stream = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::ShapeUploading);

            if (m_device_memory_radius == 0)
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_radius), sizeof(m_radius), *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void**>(m_device_memory_radius), &m_radius, sizeof(m_radius), cudaMemcpyHostToDevice, *stream));

            if (m_device_memory_center == 0)
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_center), sizeof(m_center), *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void**>(m_device_memory_center), m_center.e, sizeof(m_center), cudaMemcpyHostToDevice, *stream));

            unsigned int sbt_index = 0;
            if (m_device_memory_sbt_index == 0)
                CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_device_memory_sbt_index), sizeof(sbt_index), *stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void**>(m_device_memory_sbt_index), &sbt_index, sizeof(sbt_index), cudaMemcpyHostToDevice, *stream));

            m_upload_event->Reset(stream.Get());
            m_data_dirty = false;
        }
    }

    optix::Geometry Sphere::GetOptixGeometry() noexcept {
        optix::Geometry geo;
        geo.type               = optix::Geometry::EType::Sphere;
        geo.sphere.center.x    = m_center.x;
        geo.sphere.center.y    = m_center.y;
        geo.sphere.center.z    = m_center.z;
        geo.sphere.radius      = m_radius;
        geo.sphere.flip_normal = m_flip_normal;
        return geo;
    }

    OptixBuildInput Sphere::GetOptixBuildInput() noexcept {
        OptixBuildInput input;
        input.type        = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        input.sphereArray = {
            .vertexBuffers               = &m_device_memory_center,
            .vertexStrideInBytes         = sizeof(util::Float3),
            .numVertices                 = 1,
            .radiusBuffers               = &m_device_memory_radius,
            .radiusStrideInBytes         = sizeof(float),
            .singleRadius                = 1,
            .flags                       = &Shape::s_input_flag,
            .numSbtRecords               = 1,// num == 1, gas_sbt_index_offset will be always equal to 0
            .sbtIndexOffsetBuffer        = m_device_memory_sbt_index,
            .sbtIndexOffsetSizeInBytes   = sizeof(unsigned int),
            .sbtIndexOffsetStrideInBytes = sizeof(unsigned int),
        };

        return input;
    }
}// namespace Pupil::resource