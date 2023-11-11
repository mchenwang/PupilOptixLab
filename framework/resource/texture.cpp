#include "texture.h"
#include "texture/image.h"

#include "util/hash.h"
#include "util/id.h"

#include "cuda/stream.h"
#include "cuda/check.h"

#include "render/texture.h"

#include <unordered_map>
#include <filesystem>
#include <mutex>

namespace Pupil::resource {
    Texture::Texture(std::string_view name) noexcept : Object(name) {}
    Texture::~Texture() noexcept {}

    optix::Texture TextureInstance::GetOptixTexture() noexcept {
        auto optix_tex         = m_resource->GetOptixTexture();
        optix_tex.transform.r0 = make_float3(m_transform.matrix.r0.x, m_transform.matrix.r0.y, m_transform.matrix.r0.z);
        optix_tex.transform.r1 = make_float3(m_transform.matrix.r1.x, m_transform.matrix.r1.y, m_transform.matrix.r1.z);
        return optix_tex;
    }

    struct TextureManager::Impl {
        // use absolute path to identify texture
        std::unordered_map<std::string, uint64_t, util::StringHash, std::equal_to<>> map_path_to_id;
        // allow the same name
        std::unordered_multimap<std::string, uint64_t, util::StringHash, std::equal_to<>> map_name_to_id;
        std::unordered_map<uint64_t, util::Data<Texture>>                                 map_texture;
        std::unordered_map<uint64_t, std::string>                                         map_bitmap_id_to_path;

        util::Data<RGBTexture> default_texture = nullptr;

        util::UintIdAllocator id_allocation;

        Texture* GetTexture(uint64_t id) noexcept {
            if (auto it = map_texture.find(id); it != map_texture.end())
                return it->second.Get();
            return nullptr;
        }
    };

    TextureManager::TextureManager() noexcept {
        if (m_impl) return;
        m_impl = new Impl();

        m_impl->default_texture = std::make_unique<RGBTexture>(Texture::UserDisableTag{}, DEFAULT_TEXTURE_NAME, Float3(1.f));
    }

    TextureManager::~TextureManager() noexcept {
        Clear();
        m_impl->default_texture.Release();
        if (m_impl->map_texture.size() > 0) {
            Log::Warn("texture manager is destroyed before all memory is freed.");
        }
    }

    void TextureManager::SetTextureName(uint64_t id, std::string_view name) noexcept {
        auto tex = m_impl->GetTexture(id);
        if (tex == nullptr || tex->GetName() == name) return;
        auto range = m_impl->map_name_to_id.equal_range(tex->GetName());
        auto it    = range.first;
        for (; it != range.second; ++it) {
            if (it->second == id) break;
        }
        if (it != range.second)
            m_impl->map_name_to_id.erase(it);

        if (!tex->m_name.empty())
            Log::Info("texture rename {} to {}.", tex->m_name, name);

        tex->m_name = name;
        m_impl->map_name_to_id.emplace(name, id);
    }

    util::CountableRef<Texture> TextureManager::Register(util::Data<Texture>&& texture) noexcept {
        auto id       = m_impl->id_allocation.Allocate();
        auto name     = texture->GetName();
        texture->m_id = id;

        auto ref = texture.GetRef();
        m_impl->map_texture.emplace(id, std::move(texture));

        m_impl->map_name_to_id.emplace(name, id);
        return ref;
    }

    util::CountableRef<Texture> TextureManager::Clone(const util::CountableRef<Texture>& texture) noexcept {
        return Register(util::Data<Texture>(texture->Clone()));
    }

    util::CountableRef<Texture> TextureManager::LoadTextureFromFile(std::string_view path, bool sRGB, std::string_view name) noexcept {
        if (auto it = m_impl->map_path_to_id.find(path); it != m_impl->map_path_to_id.end()) {
            // Log::Info("texture reuse [{}].", path);
            return m_impl->map_texture.at(it->second).GetRef();
        }

        Image image;
        if (Image::Load(path.data(), image, sRGB)) {
            std::string         tex_name = name.empty() ? std::filesystem::path(path).stem().string() : std::string{name};
            util::Data<Texture> tex      = std::make_unique<Bitmap>(Texture::UserDisableTag{}, tex_name, image.w, image.h, image.data, Texture::EAddressMode::Wrap, Texture::EFilterMode::Linear);
            auto                ref      = Register(std::move(tex));

            m_impl->map_bitmap_id_to_path[ref->GetId()] = path;
            m_impl->map_path_to_id[std::string{path}]   = ref->GetId();

            delete[] image.data;
            return ref;
        }

        auto color = m_impl->default_texture->GetColor();
        Log::Warn("[{}] will be replaced by default RGBTexture({}, {}, {}).", path, color.x, color.y, color.z);
        return m_impl->default_texture.GetRef();
    }

    std::vector<const Texture*> TextureManager::GetTexture(std::string_view name) noexcept {
        std::vector<const Texture*> textures;

        if (name == DEFAULT_TEXTURE_NAME) {
            textures.push_back(m_impl->default_texture.Get());
        } else {
            auto range = m_impl->map_name_to_id.equal_range(name);
            for (auto it = range.first; it != range.second; ++it) {
                textures.push_back(m_impl->map_texture.at(it->second).Get());
            }
        }

        return textures;
    }

    util::CountableRef<Texture> TextureManager::GetTexture(uint64_t id) noexcept {
        if (auto it = m_impl->map_texture.find(id); it != m_impl->map_texture.end())
            return it->second.GetRef();

        Log::Warn("texture [{}] does not exist.", id);
        return m_impl->default_texture.GetRef();
    }

    std::vector<const Texture*> TextureManager::GetTextures() const noexcept {
        std::vector<const Texture*> textures;
        textures.reserve(m_impl->map_texture.size());
        for (auto&& [id, tex] : m_impl->map_texture)
            textures.push_back(tex.Get());
        return textures;
    }

    void TextureManager::Clear() noexcept {
        for (auto it = m_impl->map_texture.begin(); it != m_impl->map_texture.end();) {
            if (it->second.GetRefCount() == 0) {
                auto id = it->second->GetId();
                if (auto path = m_impl->map_bitmap_id_to_path.find(id);
                    path != m_impl->map_bitmap_id_to_path.end()) {
                    m_impl->map_path_to_id.erase(path->second);
                    m_impl->map_bitmap_id_to_path.erase(path);
                }

                auto range      = m_impl->map_name_to_id.equal_range(it->second->GetName());
                auto name_id_it = range.first;
                for (; name_id_it != range.second; ++name_id_it) {
                    if (name_id_it->second == id) break;
                }
                if (name_id_it != range.second)
                    m_impl->map_name_to_id.erase(name_id_it);

                m_impl->id_allocation.Recycle(id);

                it = m_impl->map_texture.erase(it);
            } else
                ++it;
        }
    }
}// namespace Pupil::resource