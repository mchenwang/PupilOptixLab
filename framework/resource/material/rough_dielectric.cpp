#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> RoughDielectric::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<RoughDielectric>(UserDisableTag{}, name));
    }

    RoughDielectric::RoughDielectric(UserDisableTag, std::string_view name) noexcept
        : Material(name),
          m_int_ior(1.5046f), m_ext_ior(1.000277f) {
        m_alpha                  = RGBTexture::Make(Float3(0.1f), m_name + " alpha");
        m_specular_reflectance   = RGBTexture::Make(Float3(1.f), m_name + " specular reflectance");
        m_specular_transmittance = RGBTexture::Make(Float3(1.f), m_name + " specular transmittance");
    }

    RoughDielectric::~RoughDielectric() noexcept {
    }

    void* RoughDielectric::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone       = new RoughDielectric(UserDisableTag{}, m_name);
        clone->m_int_ior = m_int_ior;
        clone->m_ext_ior = m_ext_ior;
        clone->m_alpha.SetTexture(tex_mngr->Clone(m_alpha));
        clone->m_alpha.SetTransform(m_alpha.GetTransform());
        clone->m_specular_reflectance.SetTexture(tex_mngr->Clone(m_specular_reflectance));
        clone->m_specular_reflectance.SetTransform(m_specular_reflectance.GetTransform());
        clone->m_specular_transmittance.SetTexture(tex_mngr->Clone(m_specular_transmittance));
        clone->m_specular_transmittance.SetTransform(m_specular_transmittance.GetTransform());
        return clone;
    }

    uint64_t RoughDielectric::GetMemorySizeInByte() const noexcept {
        return m_alpha->GetMemorySizeInByte() +
               m_specular_reflectance->GetMemorySizeInByte() +
               m_specular_transmittance->GetMemorySizeInByte() +
               sizeof(float) * 2;
    }

    void RoughDielectric::SetIntIOR(float ior) noexcept {
        m_int_ior = ior;
    }

    void RoughDielectric::SetExtIOR(float ior) noexcept {
        m_ext_ior = ior;
    }

    void RoughDielectric::SetAlpha(const Float3& alpha) noexcept {
        m_alpha.SetTexture(RGBTexture::Make(alpha, m_alpha->GetName()));
    }

    void RoughDielectric::SetSpecularReflectance(const Float3& reflectance) noexcept {
        m_specular_reflectance.SetTexture(RGBTexture::Make(reflectance, m_specular_reflectance->GetName()));
    }

    void RoughDielectric::SetSpecularTransmittance(const Float3& transmittance) noexcept {
        m_specular_transmittance.SetTexture(RGBTexture::Make(transmittance, m_specular_transmittance->GetName()));
    }

    void RoughDielectric::SetAlpha(const TextureInstance& alpha) noexcept {
        m_alpha = alpha;
    }

    void RoughDielectric::SetSpecularReflectance(const TextureInstance& reflectance) noexcept {
        m_specular_reflectance = reflectance;
    }

    void RoughDielectric::SetSpecularTransmittance(const TextureInstance& transmittance) noexcept {
        m_specular_transmittance = transmittance;
    }

    void RoughDielectric::UploadToCuda() noexcept {
        m_alpha->UploadToCuda();
        m_specular_reflectance->UploadToCuda();
        m_specular_transmittance->UploadToCuda();
    }

    optix::Material RoughDielectric::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided                                = false;
        mat.type                                    = EMatType::RoughDielectric;
        mat.rough_dielectric.eta                    = m_int_ior / m_ext_ior;
        mat.rough_dielectric.alpha                  = m_alpha.GetOptixTexture();
        mat.rough_dielectric.specular_reflectance   = m_specular_reflectance.GetOptixTexture();
        mat.rough_dielectric.specular_transmittance = m_specular_transmittance.GetOptixTexture();

        return mat;
    }
}// namespace Pupil::resource