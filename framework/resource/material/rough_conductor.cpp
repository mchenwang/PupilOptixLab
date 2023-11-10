#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> RoughConductor::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<RoughConductor>(UserDisableTag{}, name));
    }

    RoughConductor::RoughConductor(UserDisableTag, std::string_view name) noexcept
        : Material(name) {
        m_alpha                = RGBTexture::Make(util::Float3(0.1f), m_name + " alpha");
        m_eta                  = RGBTexture::Make(util::Float3(0.f), m_name + " eta");
        m_k                    = RGBTexture::Make(util::Float3(1.f), m_name + " k");
        m_specular_reflectance = RGBTexture::Make(util::Float3(1.f), m_name + " specular reflectance");
    }

    RoughConductor::~RoughConductor() noexcept {
    }

    void* RoughConductor::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone = new RoughConductor(UserDisableTag{}, m_name);
        clone->m_alpha.SetTexture(tex_mngr->Clone(m_alpha));
        clone->m_alpha.SetTransform(m_alpha.GetTransform());
        clone->m_eta.SetTexture(tex_mngr->Clone(m_eta));
        clone->m_eta.SetTransform(m_eta.GetTransform());
        clone->m_k.SetTexture(tex_mngr->Clone(m_k));
        clone->m_k.SetTransform(m_k.GetTransform());
        clone->m_specular_reflectance.SetTexture(tex_mngr->Clone(m_specular_reflectance));
        clone->m_specular_reflectance.SetTransform(m_specular_reflectance.GetTransform());
        return clone;
    }

    uint64_t RoughConductor::GetMemorySizeInByte() const noexcept {
        return m_alpha->GetMemorySizeInByte() +
               m_eta->GetMemorySizeInByte() +
               m_k->GetMemorySizeInByte() +
               m_specular_reflectance->GetMemorySizeInByte();
    }

    void RoughConductor::SetAlpha(const util::Float3& alpha) noexcept {
        m_alpha.SetTexture(RGBTexture::Make(alpha, m_alpha->GetName()));
    }

    void RoughConductor::SetEta(const util::Float3& eta) noexcept {
        m_eta.SetTexture(RGBTexture::Make(eta, m_eta->GetName()));
    }

    void RoughConductor::SetK(const util::Float3& k) noexcept {
        m_k.SetTexture(RGBTexture::Make(k, m_k->GetName()));
    }

    void RoughConductor::SetSpecularReflectance(const util::Float3& reflectance) noexcept {
        m_specular_reflectance.SetTexture(RGBTexture::Make(reflectance, m_specular_reflectance->GetName()));
    }

    void RoughConductor::SetAlpha(const TextureInstance& alpha) noexcept {
        m_alpha = alpha;
    }

    void RoughConductor::SetEta(const TextureInstance& eta) noexcept {
        m_eta = eta;
    }

    void RoughConductor::SetK(const TextureInstance& k) noexcept {
        m_k = k;
    }

    void RoughConductor::SetSpecularReflectance(const TextureInstance& reflectance) noexcept {
        m_specular_reflectance = reflectance;
    }

    void RoughConductor::UploadToCuda() noexcept {
        m_alpha->UploadToCuda();
        m_eta->UploadToCuda();
        m_k->UploadToCuda();
        m_specular_reflectance->UploadToCuda();
    }

    optix::Material RoughConductor::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided                             = false;
        mat.type                                 = EMatType::RoughConductor;
        mat.rough_conductor.alpha                = m_alpha.GetOptixTexture();
        mat.rough_conductor.eta                  = m_eta.GetOptixTexture();
        mat.rough_conductor.k                    = m_k.GetOptixTexture();
        mat.rough_conductor.specular_reflectance = m_specular_reflectance.GetOptixTexture();

        return mat;
    }
}// namespace Pupil::resource