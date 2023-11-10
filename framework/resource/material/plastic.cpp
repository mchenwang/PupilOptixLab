#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Plastic::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Plastic>(UserDisableTag{}, name));
    }

    Plastic::Plastic(UserDisableTag, std::string_view name) noexcept
        : Material(name),
          m_int_ior(1.5046f), m_ext_ior(1.000277f), m_nonlinear(false) {
        m_diffuse_reflectance  = RGBTexture::Make(util::Float3(1.f), m_name + " diffuse reflectance");
        m_specular_reflectance = RGBTexture::Make(util::Float3(1.f), m_name + " specular reflectance");
    }

    Plastic::~Plastic() noexcept {
    }

    void* Plastic::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone         = new Plastic(UserDisableTag{}, m_name);
        clone->m_int_ior   = m_int_ior;
        clone->m_ext_ior   = m_ext_ior;
        clone->m_nonlinear = m_nonlinear;
        clone->m_diffuse_reflectance.SetTexture(tex_mngr->Clone(m_diffuse_reflectance));
        clone->m_diffuse_reflectance.SetTransform(m_diffuse_reflectance.GetTransform());
        clone->m_specular_reflectance.SetTexture(tex_mngr->Clone(m_specular_reflectance));
        clone->m_specular_reflectance.SetTransform(m_specular_reflectance.GetTransform());
        return clone;
    }

    uint64_t Plastic::GetMemorySizeInByte() const noexcept {
        return sizeof(float) * 2 + sizeof(bool) +
               m_diffuse_reflectance->GetMemorySizeInByte() +
               m_specular_reflectance->GetMemorySizeInByte();
    }

    void Plastic::SetIntIOR(float ior) noexcept {
        m_int_ior = ior;
    }

    void Plastic::SetExtIOR(float ior) noexcept {
        m_ext_ior = ior;
    }

    void Plastic::SetLinear(float is_linear) noexcept {
        m_nonlinear = !is_linear;
    }

    void Plastic::SetDiffuseReflectance(const util::Float3& diffuse) noexcept {
        m_diffuse_reflectance.SetTexture(RGBTexture::Make(diffuse, m_diffuse_reflectance->GetName()));
    }

    void Plastic::SetSpecularReflectance(const util::Float3& specular) noexcept {
        m_specular_reflectance.SetTexture(RGBTexture::Make(specular, m_specular_reflectance->GetName()));
    }

    void Plastic::SetDiffuseReflectance(const TextureInstance& diffuse) noexcept {
        m_diffuse_reflectance = diffuse;
    }

    void Plastic::SetSpecularReflectance(const TextureInstance& specular) noexcept {
        m_specular_reflectance = specular;
    }

    void Plastic::UploadToCuda() noexcept {
        m_diffuse_reflectance->UploadToCuda();
        m_specular_reflectance->UploadToCuda();
    }

    optix::Material Plastic::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided                     = false;
        mat.type                         = EMatType::Plastic;
        mat.plastic.eta                  = m_int_ior / m_ext_ior;
        mat.plastic.nonlinear            = m_nonlinear;
        mat.plastic.diffuse_reflectance  = m_diffuse_reflectance.GetOptixTexture();
        mat.plastic.specular_reflectance = m_specular_reflectance.GetOptixTexture();

        auto diffuse  = m_diffuse_reflectance->GetPixelAverage();
        auto specular = m_specular_reflectance->GetPixelAverage();

        float diffuse_luminance                = optix::GetLuminance(make_float3(diffuse.r, diffuse.g, diffuse.b));
        float specular_luminance               = optix::GetLuminance(make_float3(specular.r, specular.g, specular.b));
        mat.plastic.m_specular_sampling_weight = specular_luminance / (specular_luminance + diffuse_luminance);

        mat.plastic.m_int_fdr = optix::material::fresnel::DiffuseReflectance(1.f / mat.plastic.eta);
        return mat;
    }
}// namespace Pupil::resource