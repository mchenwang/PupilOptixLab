#include "material.h"

#include "util/log.h"
#include "util/id.h"
#include "util/hash.h"

#include <unordered_map>

namespace Pupil::resource {
    Material::Material(std::string_view name) noexcept : Object(name) {}
    Material::~Material() noexcept {}

    struct MaterialManager::Impl {
        // allow the same name
        std::unordered_multimap<std::string, uint64_t, util::StringHash, std::equal_to<>> map_name_to_id;
        std::unordered_map<uint64_t, util::Data<Material>>                                map_material;

        util::Data<Material> default_material = nullptr;

        util::UintIdAllocator id_allocation;

        Material* GetMaterial(uint64_t id) noexcept {
            if (auto it = map_material.find(id); it != map_material.end())
                return it->second.Get();
            return nullptr;
        }
    };

    MaterialManager::MaterialManager() noexcept {
        if (m_impl) return;
        m_impl = new Impl();

        m_impl->default_material = std::make_unique<Diffuse>(Material::UserDisableTag{}, DEFAULT_MATERIAL_NAME);
    }

    MaterialManager::~MaterialManager() noexcept {
        Clear();
        m_impl->default_material.Release();
        if (m_impl->map_material.size() > 0) {
            Log::Warn("material manager is destroyed before all memory is freed.");
        }
    }

    void MaterialManager::SetMaterialName(uint64_t id, std::string_view name) noexcept {
        auto tex = m_impl->GetMaterial(id);
        if (tex == nullptr || tex->GetName() == name) return;
        auto range = m_impl->map_name_to_id.equal_range(tex->GetName());
        auto it    = range.first;
        for (; it != range.second; ++it) {
            if (it->second == id) break;
        }
        if (it != range.second)
            m_impl->map_name_to_id.erase(it);

        tex->m_name = name;
        m_impl->map_name_to_id.emplace(name, id);
    }

    util::CountableRef<Material> MaterialManager::Register(util::Data<Material>&& material) noexcept {
        auto id        = m_impl->id_allocation.Allocate();
        auto name      = material->GetName();
        material->m_id = id;

        auto ref = material.GetRef();
        m_impl->map_material.emplace(id, std::move(material));

        m_impl->map_name_to_id.emplace(name, id);
        return ref;
    }

    util::CountableRef<Material> MaterialManager::Clone(const util::CountableRef<Material>& material) noexcept {
        return Register(util::Data<Material>(material->Clone()));
    }

    std::vector<const Material*> MaterialManager::GetMaterial(std::string_view name) noexcept {
        std::vector<const Material*> materials;

        if (name == DEFAULT_MATERIAL_NAME) {
            materials.push_back(m_impl->default_material.Get());
        } else {
            auto range = m_impl->map_name_to_id.equal_range(name);
            for (auto it = range.first; it != range.second; ++it) {
                materials.push_back(m_impl->map_material.at(it->second).Get());
            }
        }

        return materials;
    }

    util::CountableRef<Material> MaterialManager::GetMaterial(uint64_t id) noexcept {
        if (auto it = m_impl->map_material.find(id); it != m_impl->map_material.end())
            return it->second.GetRef();

        Log::Warn("material [{}] does not exist.", id);
        return m_impl->default_material.GetRef();
    }

    std::vector<const Material*> MaterialManager::GetMaterials() const noexcept {
        std::vector<const Material*> materials;
        materials.reserve(m_impl->map_material.size());
        for (auto&& [id, tex] : m_impl->map_material)
            materials.push_back(tex.Get());
        return materials;
    }

    void Clear(auto& map, auto& name_to_id, auto& id_allocation) noexcept {
        for (auto it = map.begin(); it != map.end();) {
            if (it->second.GetRefCount() == 0) {
                auto id = it->second->GetId();

                auto range      = name_to_id.equal_range(it->second->GetName());
                auto name_id_it = range.first;
                for (; name_id_it != range.second; ++name_id_it) {
                    if (name_id_it->second == id) break;
                }
                if (name_id_it != range.second)
                    name_to_id.erase(name_id_it);

                id_allocation.Recycle(id);

                it = map.erase(it);
            } else
                ++it;
        }
    }

    void MaterialManager::Clear() noexcept {
        // sigle layer material
        resource::Clear(m_impl->map_material, m_impl->map_name_to_id, m_impl->id_allocation);
        // inner material in two-sided material
        resource::Clear(m_impl->map_material, m_impl->map_name_to_id, m_impl->id_allocation);
    }
}// namespace Pupil::resource