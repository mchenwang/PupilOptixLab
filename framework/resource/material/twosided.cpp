#include "resource/material.h"
#include "render/material.h"

namespace Pupil::resource {
    util::CountableRef<Material> Twosided::Make(std::string_view name) noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();
        return mat_mngr->Register(std::make_unique<Twosided>(UserDisableTag{}, name));
    }

    Twosided::Twosided(UserDisableTag, std::string_view name) noexcept
        : Material(name) {
        m_inner_material = Diffuse::Make(Float3(1.f), m_name + " diffuse");
    }

    Twosided::~Twosided() noexcept {
        m_inner_material.Reset();
    }

    void* Twosided::Clone() const noexcept {
        auto mat_mngr = util::Singleton<MaterialManager>::instance();

        auto clone              = new Twosided(UserDisableTag{}, m_name);
        clone->m_inner_material = mat_mngr->Clone(m_inner_material);
        return clone;
    }

    uint64_t Twosided::GetMemorySizeInByte() const noexcept {
        return m_inner_material->GetMemorySizeInByte();
    }

    void Twosided::UploadToCuda() noexcept {
        m_inner_material->UploadToCuda();
    }

    optix::Material Twosided::GetOptixMaterial() noexcept {
        optix::Material mat = m_inner_material->GetOptixMaterial();
        mat.twosided        = true;
        return mat;
    }
}// namespace Pupil::resource