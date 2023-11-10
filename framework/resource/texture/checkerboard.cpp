#include "../texture.h"
#include "render/texture.h"

namespace Pupil::resource {

    util::CountableRef<Texture> CheckerboardTexture::Make(std::string_view name) noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();
        return tex_mngr->Register(std::make_unique<CheckerboardTexture>(UserDisableTag{}, name, util::Float3(0.2f), util::Float3(0.8f)));
    }

    util::CountableRef<Texture> CheckerboardTexture::Make(const util::Float3& c1, const util::Float3& c2, std::string_view name) noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();
        return tex_mngr->Register(std::make_unique<CheckerboardTexture>(UserDisableTag{}, name, c1, c2));
    }

    CheckerboardTexture::CheckerboardTexture(UserDisableTag,
                                             std::string_view    name,
                                             const util::Float3& c1,
                                             const util::Float3& c2) noexcept
        : Texture(name),
          m_checkerborad_color1(c1),
          m_checkerborad_color2(c2) {
    }

    void* CheckerboardTexture::Clone() const noexcept {
        auto clone = new CheckerboardTexture(UserDisableTag{}, m_name, m_checkerborad_color1, m_checkerborad_color2);
        return clone;
    }

    uint64_t CheckerboardTexture::GetMemorySizeInByte() const noexcept {
        return sizeof(float) * 3 * 2;
    }

    optix::Texture CheckerboardTexture::GetOptixTexture() noexcept {
        optix::Texture tex;
        tex.type     = optix::Texture::Checkerboard;
        tex.patch1.x = m_checkerborad_color1.x;
        tex.patch1.y = m_checkerborad_color1.y;
        tex.patch1.z = m_checkerborad_color1.z;
        tex.patch2.x = m_checkerborad_color2.x;
        tex.patch2.y = m_checkerborad_color2.y;
        tex.patch2.z = m_checkerborad_color2.z;
        return tex;
    }

    util::Float3 CheckerboardTexture::GetPixelAverage() const noexcept {
        float r = m_checkerborad_color1.x + m_checkerborad_color2.x;
        float g = m_checkerborad_color1.y + m_checkerborad_color2.y;
        float b = m_checkerborad_color1.z + m_checkerborad_color2.z;
        return util::Float3(r, g, b) * 0.5f;
    }
}// namespace Pupil::resource