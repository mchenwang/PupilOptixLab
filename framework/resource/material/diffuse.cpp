#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Diffuse::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Diffuse>(UserDisableTag{}, name));
    }

    util::CountableRef<Material> Diffuse::Make(const util::Float3& c, std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Diffuse>(UserDisableTag{}, name, c));
    }

    Diffuse::Diffuse(UserDisableTag, std::string_view name, const util::Float3& c) noexcept
        : Material(name) {
        m_reflectance = RGBTexture::Make(c, m_name + " reflectance");
    }

    Diffuse::~Diffuse() noexcept {
    }

    void* Diffuse::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone = new Diffuse(UserDisableTag{}, m_name);
        clone->m_reflectance.SetTexture(tex_mngr->Clone(m_reflectance));
        clone->m_reflectance.SetTransform(m_reflectance.GetTransform());

        return clone;
    }

    uint64_t Diffuse::GetMemorySizeInByte() const noexcept {
        return m_reflectance->GetMemorySizeInByte();
    }

    void Diffuse::SetReflectance(const util::Float3& color) noexcept {
        m_reflectance.SetTexture(RGBTexture::Make(color, m_reflectance->GetName()));
    }

    void Diffuse::SetReflectance(const TextureInstance& reflectance) noexcept {
        m_reflectance = reflectance;
    }

    void Diffuse::UploadToCuda() noexcept {
        m_reflectance->UploadToCuda();
    }

    optix::Material Diffuse::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided            = false;
        mat.type                = EMatType::Diffuse;
        mat.diffuse.reflectance = m_reflectance.GetOptixTexture();
        return mat;
    }
}// namespace Pupil::resource