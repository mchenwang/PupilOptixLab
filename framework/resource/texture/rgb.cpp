#include "../texture.h"
#include "render/texture.h"

namespace Pupil::resource {
    util::CountableRef<Texture> RGBTexture::Make(const Float3& c, std::string_view name) noexcept {
        auto tex_mngr = util::Singleton<TextureManager>::instance();
        return tex_mngr->Register(std::make_unique<RGBTexture>(UserDisableTag{}, name, c));
    }

    RGBTexture::RGBTexture(UserDisableTag, std::string_view name, const Float3& c) noexcept
        : Texture(name),
          m_color(c) {
    }

    void* RGBTexture::Clone() const noexcept {
        auto clone = new RGBTexture(UserDisableTag{}, m_name, m_color);
        return clone;
    }

    uint64_t RGBTexture::GetMemorySizeInByte() const noexcept {
        return sizeof(float) * 3;
    }

    optix::Texture RGBTexture::GetOptixTexture() noexcept {
        optix::Texture tex;
        tex.type  = optix::Texture::RGB;
        tex.rgb.x = m_color.x;
        tex.rgb.y = m_color.y;
        tex.rgb.z = m_color.z;
        return tex;
    }

    Float3 RGBTexture::GetPixelAverage() const noexcept {
        return m_color;
    }
}// namespace Pupil::resource