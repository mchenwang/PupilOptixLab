#include "scene/emitter.h"

namespace Pupil {
    SphereEmitter::SphereEmitter(const util::CountableRef<resource::Sphere>& shape,
                                 const util::Transform&                      transform,
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

    void SphereEmitter::SetTransform(const util::Transform& trans) noexcept {
        m_transform = trans;

        auto o = m_shape->GetCenter();
        auto p = util::Float3(o.x + m_shape->GetRadius(), o.y, o.z);

        o = util::Transform::TransformPoint(o, m_transform.matrix);
        p = util::Transform::TransformPoint(p, m_transform.matrix);

        m_temp_center = make_float3(o.x, o.y, o.z);
        m_temp_radius = length(m_temp_center - make_float3(p.x, p.y, p.z));
        m_temp_area   = 4 * 3.14159265358979323846f * m_temp_radius * m_temp_radius;
    }

}// namespace Pupil