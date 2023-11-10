#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Hair::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Hair>(UserDisableTag{}, name));
    }

    Hair::Hair(UserDisableTag, std::string_view name) noexcept
        : Material(name),
          m_beta_m(0.3f), m_beta_n(0.3f), m_alpha(0.f) {
        m_sigma_a = RGBTexture::Make(util::Float3(0.06f, 0.1f, 0.2f), m_name + " sigma_a");
    }

    Hair::~Hair() noexcept {
    }

    void* Hair::Clone() const noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();

        auto clone      = new Hair(UserDisableTag{}, m_name);
        clone->m_beta_m = m_beta_m;
        clone->m_beta_n = m_beta_n;
        clone->m_alpha  = m_alpha;
        clone->m_sigma_a.SetTexture(tex_mngr->Clone(m_sigma_a));
        clone->m_sigma_a.SetTransform(m_sigma_a.GetTransform());
        return clone;
    }

    uint64_t Hair::GetMemorySizeInByte() const noexcept {
        return sizeof(float) * 10 +
               m_sigma_a->GetMemorySizeInByte();
    }

    void Hair::SetBetaM(float beta_m) noexcept {
        m_beta_m = beta_m;
    }

    void Hair::SetBetaN(float beta_n) noexcept {
        m_beta_n = beta_n;
    }

    void Hair::SetAlpha(float alpha) noexcept {
        m_alpha = alpha;
    }

    void Hair::SetSigmaA(const util::Float3& sigma_a) noexcept {
        m_sigma_a.SetTexture(RGBTexture::Make(sigma_a, m_sigma_a->GetName()));
    }

    void Hair::SetSigmaA(const TextureInstance& sigma_a) noexcept {
        m_sigma_a = sigma_a;
    }

    void Hair::UploadToCuda() noexcept {
        m_sigma_a->UploadToCuda();
    }

    optix::Material Hair::GetOptixMaterial() noexcept {
        optix::Material mat;
        mat.twosided = false;
        mat.type     = EMatType::Hair;

        util::Float3 v, sin_2k_alpha, cos_2k_alpha;

        v.x = pow(0.726f * m_beta_m + 0.812f * pow(m_beta_m, 2) + 3.7f * pow(m_beta_m, 20.f), 2);
        v.y = 0.25f * v.x;
        v.z = 4 * v.x;

        float s;
        s = 0.626657069f *
            (0.265f * m_beta_n + 1.194f * pow(m_beta_n, 2) + 5.372f * pow(m_beta_n, 22.f));

        sin_2k_alpha.x = sin(m_alpha);
        cos_2k_alpha.x = sqrt(1 - pow(sin_2k_alpha.x, 2));
        sin_2k_alpha.y = 2 * cos_2k_alpha.x * sin_2k_alpha.x;
        cos_2k_alpha.y = pow(cos_2k_alpha.x, 2) - pow(sin_2k_alpha.x, 2);
        sin_2k_alpha.z = 2 * cos_2k_alpha.y * sin_2k_alpha.y;
        cos_2k_alpha.z = pow(cos_2k_alpha.y, 2) - pow(sin_2k_alpha.y, 2);

        mat.hair.azimuthal_s    = s;
        mat.hair.longitudinal_v = make_float3(v.x, v.y, v.z);
        mat.hair.sin_2k_alpha   = make_float3(sin_2k_alpha.x, sin_2k_alpha.y, sin_2k_alpha.z);
        mat.hair.cos_2k_alpha   = make_float3(cos_2k_alpha.x, cos_2k_alpha.y, cos_2k_alpha.z);
        mat.hair.sigma_a        = m_sigma_a.GetOptixTexture();
        return mat;
    }
}// namespace Pupil::resource