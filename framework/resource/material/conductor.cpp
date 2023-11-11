#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Conductor::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Conductor>(UserDisableTag{}, name));
    }

    Conductor::Conductor(UserDisableTag, std::string_view name) noexcept
        : Material(name) {
        m_eta                  = RGBTexture::Make(Float3(0.f), m_name + " eta");
        m_k                    = RGBTexture::Make(Float3(1.f), m_name + " k");
        m_specular_reflectance = RGBTexture::Make(Float3(1.f), m_name + " specular reflectance");
    }

    Conductor::~Conductor() noexcept {
    }

    void* Conductor::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone = new Conductor(UserDisableTag{}, m_name);
        clone->m_eta.SetTexture(tex_mngr->Clone(m_eta));
        clone->m_eta.SetTransform(m_eta.GetTransform());
        clone->m_k.SetTexture(tex_mngr->Clone(m_k));
        clone->m_k.SetTransform(m_k.GetTransform());
        clone->m_specular_reflectance.SetTexture(tex_mngr->Clone(m_specular_reflectance));
        clone->m_specular_reflectance.SetTransform(m_specular_reflectance.GetTransform());
        return clone;
    }

    uint64_t Conductor::GetMemorySizeInByte() const noexcept {
        return m_eta->GetMemorySizeInByte() +
               m_k->GetMemorySizeInByte() +
               m_specular_reflectance->GetMemorySizeInByte();
    }

    void Conductor::SetEta(const Float3& eta) noexcept {
        m_eta.SetTexture(RGBTexture::Make(eta, m_eta->GetName()));
    }

    void Conductor::SetK(const Float3& k) noexcept {
        m_k.SetTexture(RGBTexture::Make(k, m_k->GetName()));
    }

    void Conductor::SetSpecularReflectance(const Float3& reflectance) noexcept {
        m_specular_reflectance.SetTexture(RGBTexture::Make(reflectance, m_specular_reflectance->GetName()));
    }

    void Conductor::SetEta(const TextureInstance& eta) noexcept {
        m_eta = eta;
    }

    void Conductor::SetK(const TextureInstance& k) noexcept {
        m_k = k;
    }

    void Conductor::SetSpecularReflectance(const TextureInstance& reflectance) noexcept {
        m_specular_reflectance = reflectance;
    }

    void Conductor::UploadToCuda() noexcept {
        m_eta->UploadToCuda();
        m_k->UploadToCuda();
        m_specular_reflectance->UploadToCuda();
    }

    optix::Material Conductor::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided                       = false;
        mat.type                           = EMatType::Conductor;
        mat.conductor.eta                  = m_eta.GetOptixTexture();
        mat.conductor.k                    = m_k.GetOptixTexture();
        mat.conductor.specular_reflectance = m_specular_reflectance.GetOptixTexture();

        return mat;
    }
}// namespace Pupil::resource