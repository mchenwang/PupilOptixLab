#include "scene/emitter.h"
#include "cuda/util.h"

namespace Pupil {
    SphereEmitter::SphereEmitter(const util::CountableRef<resource::Sphere>& shape,
                                 const Transform&                            transform,
                                 const resource::TextureInstance&            radiance) noexcept
        : Emitter(radiance, transform), m_shape(shape) {
        SetTransform(m_transform);
    }

    SphereEmitter::~SphereEmitter() noexcept {
        m_shape.Reset();
    }

    void SphereEmitter::UploadToCuda() noexcept {
        m_radiance->UploadToCuda();
    }

    optix::Emitter SphereEmitter::GetOptixEmitter() noexcept {
        optix::Emitter emitter;
        emitter.type              = optix::EEmitterType::Sphere;
        emitter.sphere.geo.center = m_temp_center;
        emitter.sphere.geo.radius = m_temp_radius;
        emitter.sphere.area       = m_temp_area;
        emitter.sphere.radiance   = m_radiance.GetOptixTexture();

        return emitter;
    }

    void SphereEmitter::SetTransform(const Transform& trans) noexcept {
        m_transform = trans;

        auto o = m_shape->GetCenter();
        auto p = Float3(o.x + m_shape->GetRadius(), o.y, o.z);

        o = m_transform * o;
        p = m_transform * p;

        m_temp_center = cuda::MakeFloat3(o);
        m_temp_radius = Lengthf(o - p);
        m_temp_area   = 4 * Pupil::PI * m_temp_radius * m_temp_radius;
    }

}// namespace Pupil