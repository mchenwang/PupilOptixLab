#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Dielectric::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Dielectric>(UserDisableTag{}, name));
    }

    Dielectric::Dielectric(UserDisableTag, std::string_view name) noexcept
        : Material(name),
          m_int_ior(1.5046f), m_ext_ior(1.000277f) {
        m_specular_reflectance   = RGBTexture::Make(util::Float3(1.f), m_name + " specular reflectance");
        m_specular_transmittance = RGBTexture::Make(util::Float3(1.f), m_name + " specular transmittance");
    }

    Dielectric::~Dielectric() noexcept {
    }

    void* Dielectric::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone       = new Dielectric(UserDisableTag{}, m_name);
        clone->m_int_ior = m_int_ior;
        clone->m_ext_ior = m_ext_ior;
        clone->m_specular_reflectance.SetTexture(tex_mngr->Clone(m_specular_reflectance));
        clone->m_specular_reflectance.SetTransform(m_specular_reflectance.GetTransform());
        clone->m_specular_transmittance.SetTexture(tex_mngr->Clone(m_specular_transmittance));
        clone->m_specular_transmittance.SetTransform(m_specular_transmittance.GetTransform());
        return clone;
    }

    uint64_t Dielectric::GetMemorySizeInByte() const noexcept {
        return m_specular_reflectance->GetMemorySizeInByte() +
               m_specular_transmittance->GetMemorySizeInByte() +
               sizeof(float) * 2;
    }

    void Dielectric::SetIntIOR(float ior) noexcept {
        m_int_ior = ior;
    }

    void Dielectric::SetExtIOR(float ior) noexcept {
        m_ext_ior = ior;
    }

    void Dielectric::SetSpecularReflectance(const util::Float3& reflectance) noexcept {
        m_specular_reflectance.SetTexture(RGBTexture::Make(reflectance, m_specular_reflectance->GetName()));
    }

    void Dielectric::SetSpecularTransmittance(const util::Float3& transmittance) noexcept {
        m_specular_transmittance.SetTexture(RGBTexture::Make(transmittance, m_specular_transmittance->GetName()));
    }

    void Dielectric::SetSpecularReflectance(const TextureInstance& reflectance) noexcept {
        m_specular_reflectance = reflectance;
    }

    void Dielectric::SetSpecularTransmittance(const TextureInstance& transmittance) noexcept {
        m_specular_transmittance = transmittance;
    }

    void Dielectric::UploadToCuda() noexcept {
        m_specular_reflectance->UploadToCuda();
        m_specular_transmittance->UploadToCuda();
    }

    optix::Material Dielectric::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided                          = false;
        mat.type                              = EMatType::Dielectric;
        mat.dielectric.int_ior                = m_int_ior;
        mat.dielectric.ext_ior                = m_ext_ior;
        mat.dielectric.specular_reflectance   = m_specular_reflectance.GetOptixTexture();
        mat.dielectric.specular_transmittance = m_specular_transmittance.GetOptixTexture();

        return mat;
    }
}// namespace Pupil::resource