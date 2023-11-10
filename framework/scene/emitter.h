#pragma once

#include "resource/shape.h"
#include "resource/texture.h"
#include "render/emitter.h"

#include "util/data.h"
#include "util/util.h"

#include <cuda.h>

namespace Pupil {
    class Emitter {
    public:
        Emitter() noexcept = default;
        Emitter(const resource::TextureInstance& radiance, const util::Transform& transform) noexcept
            : m_radiance(radiance), m_transform(transform) {}
        virtual ~Emitter() noexcept = default;

        virtual optix::Emitter GetOptixEmitter() noexcept = 0;
        virtual bool           IsEnvironmentEmitter() const noexcept { return false; }
        virtual void           UploadToCuda() noexcept = 0;

        virtual void SetTransform(const util::Transform& trans) noexcept { m_transform = trans; };
        void         SetRadiance(const resource::TextureInstance& radiance) noexcept { m_radiance = radiance; }
        auto         GetTransform() const noexcept { return m_transform; }
        auto&        GetRadiance() const noexcept { return m_radiance; }

    protected:
        resource::TextureInstance m_radiance;
        util::Transform           m_transform;
    };

    class SphereEmitter final : public Emitter {
    public:
        SphereEmitter(const util::CountableRef<resource::Sphere>& shape,
                      const util::Transform&                      transform,
                      const resource::TextureInstance&            radiance) noexcept;
        ~SphereEmitter() noexcept;

        virtual optix::Emitter GetOptixEmitter() noexcept override;
        virtual void           SetTransform(const util::Transform& trans) noexcept override;
        virtual void           UploadToCuda() noexcept override;

        auto GetShape() const noexcept { return m_shape.Get(); }

    private:
        util::CountableRef<resource::Sphere> m_shape;

        float3 m_temp_center;
        float  m_temp_radius;
        float  m_temp_area;
    };

    class TriMeshEmitter final : public Emitter {
    public:
        TriMeshEmitter(const util::CountableRef<resource::TriangleMesh>& shape,
                       const util::Transform&                            transform,
                       const resource::TextureInstance&                  radiance) noexcept;
        ~TriMeshEmitter() noexcept;

        virtual optix::Emitter GetOptixEmitter() noexcept override;
        virtual void           SetTransform(const util::Transform& trans) noexcept override;
        virtual void           UploadToCuda() noexcept override;

        auto GetShape() const noexcept { return m_shape.Get(); }

    private:
        bool                                       m_data_dirty;
        CUdeviceptr                                m_cuda_memory;
        util::CountableRef<resource::TriangleMesh> m_shape;

        uint32_t                    m_num_face;
        std::unique_ptr<uint32_t[]> m_mesh_index;
        std::unique_ptr<float[]>    m_areas;
        uint32_t                    m_num_vertex;
        std::unique_ptr<float[]>    m_mesh_vertex_ws;
        std::unique_ptr<float[]>    m_mesh_normal_ws;
        std::unique_ptr<float[]>    m_mesh_texcoord_ls;
    };

    class EnvmapEmitter final : public Emitter {
    public:
        EnvmapEmitter(const resource::TextureInstance& radiance) noexcept;
        ~EnvmapEmitter() noexcept;

        virtual optix::Emitter GetOptixEmitter() noexcept override;
        virtual bool           IsEnvironmentEmitter() const noexcept override { return true; }
        virtual void           UploadToCuda() noexcept override;

        void SetScale(float scale) noexcept { m_scale = scale; }
        auto GetScale() const noexcept { return m_scale; }

    private:
        CUdeviceptr  m_cuda_memory;
        unsigned int m_width;
        unsigned int m_height;
        float        m_normalization;
        float        m_scale;
    };

    class ConstEmitter final : public Emitter {
    public:
        ConstEmitter(const util::Float3& radiance) noexcept;
        ~ConstEmitter() noexcept;

        virtual optix::Emitter GetOptixEmitter() noexcept override;
        virtual bool           IsEnvironmentEmitter() const noexcept override { return true; }
        virtual void           UploadToCuda() noexcept override {}
    };
}// namespace Pupil